#include <flexeva/code/frontend.hpp>

#include <clang/AST/ASTConsumer.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/AST/ParentMapContext.h>
#include <clang/Analysis/CFG.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Lex/Lexer.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <clang/Tooling/Tooling.h>
#include <clang/Tooling/CompilationDatabase.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/SHA256.h>
#include <llvm/Support/raw_ostream.h>

#include <algorithm>
#include <fstream>
#include <filesystem>
#include <regex>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace flexeva::code {
namespace {
using namespace clang;
using J = llvm::json::Value;
using O = llvm::json::Object;
using A = llvm::json::Array;

std::string encode(const J& value) {
    std::string out; llvm::raw_string_ostream stream(out); stream << value; return stream.str();
}
std::string hash(llvm::StringRef value) {
    const auto bytes = llvm::SHA256::hash(llvm::arrayRefFromStringRef(value));
    return llvm::toHex(llvm::ArrayRef<uint8_t>(bytes), true);
}
std::uint64_t id(llvm::StringRef value) {
    const auto bytes = llvm::SHA256::hash(llvm::arrayRefFromStringRef(value));
    std::uint64_t out = 0;
    for (unsigned i=0; i<8; ++i) out=(out<<8)|bytes[i];
    return out ? out : 1;
}
std::string read(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot read C++ source: " + path);
    return {std::istreambuf_iterator<char>(stream), {}};
}
// The accepted set matches the direct Maya native-record protocol. Unknown
// calls fail admission rather than silently claiming complete effects.
bool emission(llvm::StringRef name) {
    static const std::set<std::string> names = {
        "cudaMalloc","cudaFree","cudaMallocAsync","cudaFreeAsync","cudaMemsetAsync",
        "cudaMemcpyAsync","cudaStreamCreate","cudaStreamCreateWithFlags","cudaStreamCreateWithPriority",
        "cudaStreamDestroy","cudaStreamSynchronize","cudaDeviceSynchronize","cudaSetDevice",
        "cudaEventCreate","cudaEventCreateWithFlags","cudaEventDestroy","cudaEventRecord",
        "cudaEventSynchronize","cudaStreamWaitEvent","cudaLaunchKernel","cudaLaunchKernelExC",
        "cublasCreate_v2","cublasDestroy_v2","cublasSetStream_v2","cublasSetPointerMode_v2",
        "cublasSetMathMode","cublasSgemm_v2","cublasDgemm_v2","cublasCgemm_v2","cublasZgemm_v2",
        "cublasGemmEx","cublasSgemmEx","cublasGemmBatchedEx","cublasGemmStridedBatchedEx",
        "cublasSgemmStridedBatched","cublasDgemmStridedBatched","cublasCgemmStridedBatched",
        "cublasZgemmStridedBatched","ncclGetUniqueId","ncclCommInitRank","ncclCommDestroy",
        "ncclGroupStart","ncclGroupEnd","ncclSend","ncclRecv","ncclAllReduce"};
    return names.contains(name.str());
}

bool pure_math(llvm::StringRef name) {
    static const std::set<std::string> names={"sqrt","sqrtf","sqrtl","pow","powf","exp","expf","log","logf","log2","log2f","floor","floorf","ceil","ceilf","fabs","fabsf","sin","sinf","cos","cosf","tanh","tanhf"};
    return names.contains(name.str());
}

struct RefVisitor : RecursiveASTVisitor<RefVisitor> {
    std::set<const VarDecl*> refs;
    bool VisitDeclRefExpr(DeclRefExpr* expr) {
        if (const auto* var=dyn_cast<VarDecl>(expr->getDecl())) refs.insert(var->getCanonicalDecl());
        return true;
    }
};
std::set<const VarDecl*> refs(const Stmt* stmt) {
    RefVisitor visitor; if(stmt) visitor.TraverseStmt(const_cast<Stmt*>(stmt)); return visitor.refs;
}

struct DefsVisitor : RecursiveASTVisitor<DefsVisitor> {
    std::map<const VarDecl*,std::vector<const Expr*>> definitions;
    bool VisitVarDecl(VarDecl* var) {
        if(var->hasInit()) definitions[var->getCanonicalDecl()].push_back(var->getInit());
        return true;
    }
    bool VisitBinaryOperator(BinaryOperator* expr) {
        if(expr->isAssignmentOp()) if(auto* ref=dyn_cast<DeclRefExpr>(expr->getLHS()->IgnoreParenImpCasts()))
            if(auto* var=dyn_cast<VarDecl>(ref->getDecl())) definitions[var->getCanonicalDecl()].push_back(expr->getRHS());
        return true;
    }
    bool VisitUnaryOperator(UnaryOperator* expr) {
        if(expr->isIncrementDecrementOp()) if(auto* ref=dyn_cast<DeclRefExpr>(expr->getSubExpr()->IgnoreParenImpCasts()))
            if(auto* var=dyn_cast<VarDecl>(ref->getDecl())) definitions[var->getCanonicalDecl()].push_back(expr);
        return true;
    }
};

struct Analysis : RecursiveASTVisitor<Analysis> {
    ASTContext& context;
    SourceManager& sm;
    Rewriter& rewrite;
    const CppSource& request;
    CppAnalysis& output;
    std::vector<FunctionDecl*> owned;
    std::string owned_class_semantics;
    FunctionDecl* entry=nullptr;
    FunctionDecl* current=nullptr;
    DefsVisitor definitions;
    A sites, functions, cfgs, dependencies, controls, rank_observations;
    std::map<std::string,unsigned> ordinals;
    std::map<const Stmt*,std::uint64_t> statement_ids;
    std::set<const VarDecl*> rank_tainted;
    std::map<const FunctionDecl*, std::set<const FunctionDecl*>> calls;
    std::map<std::uint64_t,const FunctionDecl*> owned_call_sites;
    std::map<const VarDecl*,std::uint64_t> variable_ids;
    std::vector<std::pair<unsigned,std::string>> inserts;
    std::vector<std::tuple<unsigned,unsigned,std::string>> replacements;
    bool sharing_complete=true;
    bool local_reuse_complete=true;
    std::set<const CallExpr*> wrapped_calls;
    std::vector<std::string> share_reasons;

    Analysis(ASTContext& c, Rewriter& r, const CppSource& req, CppAnalysis& out)
        :context(c),sm(c.getSourceManager()),rewrite(r),request(req),output(out){}
    bool local(SourceLocation loc) const {return loc.isValid() && sm.isWrittenInMainFile(sm.getExpansionLoc(loc));}
    [[noreturn]] void fail(const Stmt* statement,const std::string& reason) const {
        auto loc=sm.getPresumedLoc(statement ? statement->getBeginLoc() : entry->getLocation());
        throw std::runtime_error(std::string(loc.getFilename())+":"+std::to_string(loc.getLine())+":"+
                                 std::to_string(loc.getColumn())+": "+reason);
    }
    std::string text(SourceRange range) const {
        return Lexer::getSourceText(CharSourceRange::getTokenRange(range),sm,context.getLangOpts()).str();
    }
    std::string semantic_text(const Stmt* statement) const {
        std::string result;llvm::raw_string_ostream out(result);
        statement->printPretty(out,nullptr,context.getPrintingPolicy());return out.str();
    }
    unsigned offset(SourceLocation loc) const {return sm.getFileOffset(sm.getSpellingLoc(loc));}
    unsigned end(SourceLocation loc) const {
        return offset(Lexer::getLocForEndOfToken(loc,0,sm,context.getLangOpts()));
    }
    void insert(SourceLocation loc,std::string code) {
        if(loc.isMacroID())fail(nullptr,"instrumented host control/definition cannot originate in a macro expansion");
        inserts.emplace_back(offset(loc),std::move(code));
    }
    void replace(SourceRange range,std::string code) {
        if(range.getBegin().isMacroID()||range.getEnd().isMacroID()) fail(nullptr,"cannot rewrite a macro-expanded host expression");
        replacements.emplace_back(offset(range.getBegin()),end(range.getEnd()),std::move(code));
    }
    std::uint64_t site(const Stmt* stmt,const std::string& kind,const std::string& role) {
        auto key=current->getQualifiedNameAsString()+":"+kind+":"+role;
        const auto coordinate=key+":"+std::to_string(ordinals[key]++);
        const auto value=id(coordinate);
        statement_ids[stmt]=value;
        auto loc=sm.getPresumedLoc(stmt->getBeginLoc());
        A uses;for(auto* var:refs(stmt))uses.push_back(std::to_string(variable(var)));
        sites.push_back(O{{"id",std::to_string(value)},{"kind",kind},{"symbol",current->getQualifiedNameAsString()},
                         {"line",loc.getLine()},{"column",loc.getColumn()},{"role",role},
                         {"coordinate",coordinate},{"uses",std::move(uses)},{"semantic_digest",hash(semantic_text(stmt))}});
        return value;
    }
    std::uint64_t variable(const VarDecl* var) {
        auto [found,inserted]=variable_ids.emplace(var->getCanonicalDecl(),0);
        if(inserted)found->second=id(current->getQualifiedNameAsString()+":variable:"+var->getNameAsString()+":"+std::to_string(variable_ids.size()));
        return found->second;
    }
    bool scalar(QualType type) const {
        type=type.getCanonicalType().getUnqualifiedType();
        if(type->isBooleanType()||type->isEnumeralType()) return true;
        if(type->isIntegerType()) return context.getTypeSize(type)<=64;
        return type->isFloatingType() && context.getTypeSize(type)<=64;
    }
    std::string typeName(QualType type) const {
        type=type.getCanonicalType().getUnqualifiedType();
        if(type->isBooleanType()) return "bool";
        if(type->isIntegerType()) {
            const auto bits=context.getTypeSize(type);
            if(bits!=32 && bits!=64) throw std::runtime_error("entry parameters require 32/64 bit integers");
            return std::string(type->isUnsignedIntegerType()?"std::uint":"std::int")+std::to_string(bits)+"_t";
        }
        if(type->isFloatingType() && context.getTypeSize(type)==32) return "float";
        if(type->isFloatingType() && context.getTypeSize(type)==64) return "double";
        throw std::runtime_error("entry parameters must be bool, 32/64 bit integers, float or double by value");
    }
    bool tainted(const Expr* expr) const {
        for(auto* var:refs(expr)) if(rank_tainted.contains(var)) return true;
        return false;
    }
    J expression(const Expr* original,std::set<const VarDecl*> seen={}) const {
        if(!original) return nullptr;
        const auto* expr=original->IgnoreParens();
        if(auto* cast=dyn_cast<CastExpr>(expr)) {
            const auto from=cast->getSubExpr()->getType(),to=cast->getType();
            if(from->isIntegerType()&&to->isIntegerType()&&
               (context.getTypeSize(from)!=context.getTypeSize(to)||from->isUnsignedIntegerType()!=to->isUnsignedIntegerType())) {
                Expr::EvalResult constant;
                if(cast->EvaluateAsInt(constant,context)&&constant.Val.getInt().getMinSignedBits()<=64)
                    return O{{"constant",constant.Val.getInt().getSExtValue()}};
                return nullptr;
            }
            if(cast->getCastKind()!=CK_LValueToRValue && cast->getCastKind()!=CK_NoOp && cast->getCastKind()!=CK_IntegralCast)return nullptr;
            return expression(cast->getSubExpr(),std::move(seen));
        }
        if(auto* value=dyn_cast<IntegerLiteral>(expr)) return O{{"constant",value->getValue().getLimitedValue()}};
        if(auto* value=dyn_cast<CXXBoolLiteralExpr>(expr)) return O{{"constant",value->getValue()?1:0}};
        if(auto* ref=dyn_cast<DeclRefExpr>(expr)) {
            if(auto* enumeration=dyn_cast<EnumConstantDecl>(ref->getDecl()))
                return O{{"constant",enumeration->getInitVal().getSExtValue()}};
            auto* var=dyn_cast<VarDecl>(ref->getDecl());
            if(!var) return nullptr;
            var=var->getCanonicalDecl();
            if(auto* parm=dyn_cast<ParmVarDecl>(var)) {
                if(definitions.definitions.contains(var))return nullptr;
                if(parm->getDeclContext()==entry)
                    return O{{"parameter",parm->getNameAsString()}};
                return nullptr;
            }
            if(!seen.insert(var).second) return nullptr;
            auto found=definitions.definitions.find(var);
            if(found!=definitions.definitions.end() && found->second.size()==1)
                return expression(found->second.front(),std::move(seen));
            return nullptr;
        }
        if(auto* cast=dyn_cast<ExplicitCastExpr>(expr)) return expression(cast->getSubExpr(),std::move(seen));
        if(auto* unary=dyn_cast<UnaryOperator>(expr)) {
            if(unary->getOpcode()!=UO_Minus && unary->getOpcode()!=UO_Plus && unary->getOpcode()!=UO_LNot && unary->getOpcode()!=UO_Not) return nullptr;
            if(unary->getType()->isUnsignedIntegerType()&&(unary->getOpcode()==UO_Minus||unary->getOpcode()==UO_Not))return nullptr;
            auto operand=expression(unary->getSubExpr(),seen);if(operand.kind()==J::Null)return nullptr;
            return O{{"operator",UnaryOperator::getOpcodeStr(unary->getOpcode()).str()},{"operands",A{std::move(operand)}}};
        }
        if(auto* binary=dyn_cast<BinaryOperator>(expr)) {
            if(binary->isAssignmentOp() || binary->getOpcode()==BO_Comma) return nullptr;
            if(binary->getType()->isUnsignedIntegerType() &&
               (binary->getOpcode()==BO_Add||binary->getOpcode()==BO_Sub||binary->getOpcode()==BO_Mul||binary->isShiftOp()))return nullptr;
            auto lhs=expression(binary->getLHS(),seen),rhs=expression(binary->getRHS(),seen);
            if(lhs.kind()==J::Null||rhs.kind()==J::Null)return nullptr;
            return O{{"operator",binary->getOpcodeStr().str()},{"operands",A{std::move(lhs),std::move(rhs)}}};
        }
        return nullptr;
    }
    void observation(std::uint64_t site_id,const Expr* expr,const std::string& role) {
        if(!tainted(expr)) return;
        auto formula=expression(expr);
        if(formula.kind()==J::Null) {sharing_complete=false;share_reasons.push_back("unresolved rank expression at "+std::to_string(site_id));return;}
        rank_observations.push_back(O{{"site",std::to_string(site_id)},{"role",role},{"expression",std::move(formula)}});
    }
    std::string values(const Stmt* stmt) const {
        std::vector<std::string> names;
        for(auto* var:refs(stmt)) {
            if(!scalar(var->getType()))continue;
            // An output-only scalar does not have a value before the call.
            if(!isa<ParmVarDecl>(var)&&!var->hasInit())continue;
            names.push_back(var->getNameAsString());
        }
        std::sort(names.begin(),names.end());names.erase(std::unique(names.begin(),names.end()),names.end());
        std::string out="{";for(const auto& name:names){if(out.size()>1)out+=",";out+="::flexeva::code::scalar("+name+")";}return out+"}";
    }
    std::string torch_values(const Stmt* stmt) const {
        std::vector<std::string> names;
        for(auto* var:refs(stmt)) {
            if(var->hasGlobalStorage()||(!isa<ParmVarDecl>(var)&&!var->hasInit()))continue;
            if(var->hasInit()&&offset(var->getInit()->getBeginLoc())<=offset(stmt->getBeginLoc())&&end(var->getInit()->getEndLoc())>=end(stmt->getEndLoc()))continue;
            const auto name=var->getNameAsString();
            if(!name.empty())names.push_back(name);
        }
        std::sort(names.begin(),names.end());names.erase(std::unique(names.begin(),names.end()),names.end());
        std::string out="::flexeva::code::torch_values(";
        for(const auto& name:names){if(out.back()!='(')out+=",";out+=name;}return out+")";
    }
    std::set<const VarDecl*> effects(const CallExpr* call) const {
        std::set<const VarDecl*> result;
        for(auto* var:refs(call))if(var->getType()->isPointerType())result.insert(var);
        DefsVisitor changes;changes.TraverseStmt(const_cast<CallExpr*>(call));
        for(const auto& [var,_]:changes.definitions)result.insert(var);
        return result;
    }
    void bracket(Stmt* stmt,std::uint64_t site_id,const std::string& kind) {
        (void)kind;
        const auto prefix="{ ::flexeva::code::PathScope __fx_scope_"+std::to_string(site_id)+"("+std::to_string(site_id)+"ULL); ";
        if(auto* compound=dyn_cast<CompoundStmt>(stmt)) {
            insert(Lexer::getLocForEndOfToken(compound->getLBracLoc(),0,sm,context.getLangOpts()),prefix.substr(2));
        } else {
            insert(stmt->getBeginLoc(),prefix);
            auto finish=Lexer::findLocationAfterToken(stmt->getEndLoc(),tok::semi,sm,context.getLangOpts(),true);
            if(finish.isInvalid())finish=Lexer::getLocForEndOfToken(stmt->getEndLoc(),0,sm,context.getLangOpts());
            insert(finish," }");
        }
    }
    bool enclosing_call(const CallExpr* call) const {
        const Stmt* next=call;
        while(next) {auto parents=context.getParents(*next);if(parents.empty())break;next=parents[0].get<Stmt>();
            if(const auto* parent=dyn_cast_or_null<CallExpr>(next);parent && wrapped_calls.contains(parent))return true;}
        return false;
    }
    bool VisitCallExpr(CallExpr* call) {
        if(!current||!local(call->getBeginLoc()))return true;
        auto* target=call->getDirectCallee();
        if(!target) {
            if(request.torch) {local_reuse_complete=false;sharing_complete=false;return true;}
            fail(call,"indirect C++ calls are outside the restricted host frontend");
        }

        const auto name=target->getNameAsString(), qualified=target->getQualifiedNameAsString();
        if(request.torch) {
            static const std::set<std::string> ground_required={"item","is_nonzero","nonzero","masked_select","unique","unique_consecutive","data_ptr","accessor","packed_accessor","tolist"};
            if(ground_required.contains(name) && std::find(request.ground_functions.begin(),request.ground_functions.end(),qualified)==request.ground_functions.end())
                fail(call,"data-dependent Torch result requires an explicitly registered Ground function: "+qualified);
            if(enclosing_call(call)) {
                const auto* body=target->getDefinition();
                const bool known=(body&&local(body->getLocation()))||qualified.starts_with("at::")||qualified.starts_with("torch::")||qualified.starts_with("c10::")||qualified.starts_with("std::")||pure_math(qualified);
                if(!known){local_reuse_complete=false;sharing_complete=false;share_reasons.push_back("conservative nested effect: "+qualified);}
                return true;
            }
        }
        if(request.torch && !emission(name) &&
           std::find(request.ground_functions.begin(),request.ground_functions.end(),qualified)==request.ground_functions.end()) {
            if(call->getBeginLoc().isMacroID() || call->getEndLoc().isMacroID()) {
                local_reuse_complete=false;sharing_complete=false;return true;
            }
            static const std::set<std::string> ground_required={"item","is_nonzero","nonzero","masked_select","unique","unique_consecutive","data_ptr","accessor","packed_accessor","tolist"};
            if(ground_required.contains(name))fail(call,"data-dependent Torch result requires an explicitly registered Ground function: "+qualified);
            const auto* body=target->getDefinition();
            const bool owned_body=body&&local(body->getLocation())&&!body->getLocation().isMacroID();
            const bool framework=qualified.starts_with("at::")||qualified.starts_with("torch::")||qualified.starts_with("c10::");
            const bool pure_host=qualified.starts_with("std::") && qualified.find("thread")==std::string::npos && qualified.find("async")==std::string::npos;
            const bool known=owned_body||framework||pure_host||pure_math(qualified);
            if(!known){local_reuse_complete=false;sharing_complete=false;share_reasons.push_back("conservative unknown effect: "+qualified);}
            if(owned_body)calls[current->getCanonicalDecl()].insert(body->getCanonicalDecl());
            const auto sid=site(call,owned_body?"function":"emission",qualified);
            auto* record=sites.back().getAsObject();(*record)["effect_complete"]=known;
            if(owned_body){(*record)["semantic_digest"]=hash(semantic_text(call)+semantic_text(body->getBody()));owned_call_sites[sid]=body->getCanonicalDecl();}
            (*record)["framework_effect"]=framework;
            A writes;
            // Functional outputs are assigned below by their declaration or assignment;
            // in-place/autograd/optimizer effects explicitly advance their input producers.
            const auto* method=dyn_cast<CXXMethodDecl>(target);
            const bool mutation=(method&&!method->isConst()&&name!="forward"&&name!="operator()")||name=="backward"||name=="step"||name=="zero_grad"||name=="to"||name=="push_back"||name=="emplace_back"||name=="clear"||name.ends_with("_");
            if(mutation)for(auto* var:refs(call))writes.push_back(std::to_string(variable(var)));
            const Stmt* owner=call;
            while(owner) {auto parents=context.getParents(*owner);if(parents.empty())break;
                if(const auto* variable_decl=parents[0].get<VarDecl>()) {writes.push_back(std::to_string(variable(variable_decl)));break;}
                owner=parents[0].get<Stmt>();
                if(const auto* assign=dyn_cast_or_null<BinaryOperator>(owner);assign&&assign->isAssignmentOp()) {
                    if(const auto* ref=dyn_cast<DeclRefExpr>(assign->getLHS()->IgnoreParenImpCasts()))if(const auto* var=dyn_cast<VarDecl>(ref->getDecl()))writes.push_back(std::to_string(variable(var)));break;
                }
            }
            (*record)["defines"]=std::move(writes);(*record)["state_effect"]=mutation;
            for(auto* arg:call->arguments())observation(sid,arg,"argument");
            if(auto* member=dyn_cast<CXXMemberCallExpr>(call))observation(sid,member->getImplicitObjectArgument(),"receiver");
            const auto original=text(call->getSourceRange());
            replace(call->getSourceRange()," ::flexeva::code::torch_call("+std::to_string(sid)+"ULL, [&]() -> decltype(auto) { return ("+original+"); }, "+torch_values(call)+")");
            wrapped_calls.insert(call);return true;
        }
        if(emission(name)) {
            if(qualified!=name||!target->isExternC()||(target->getDefinition()&&local(target->getDefinition()->getLocation())))
                fail(call,"CUDA effect contracts require the external global C API, not a shadowing host function");
            if(name=="cudaMemcpyAsync") {
                Expr::EvalResult direction;
                if(call->getNumArgs()!=5||!call->getArg(3)->EvaluateAsInt(direction,context)||
                   (direction.Val.getInt()!=1&&direction.Val.getInt()!=3))
                    fail(call,"host-visible/device-to-host copies require a registered Ground function; only explicit H2D and D2D copies are admitted");
            }
            if(name=="ncclCommInitRank")for(auto* argument:call->arguments())
                if(argument->HasSideEffects(context))fail(call,"communicator initialization arguments must be side-effect-free for automatic membership binding");
            const auto sid=site(call,"emission",name);
            A writes;
            for(auto* var:effects(call))writes.push_back(std::to_string(variable(var)));
            (*sites.back().getAsObject())["defines"]=std::move(writes);
            for(unsigned i=0;i<call->getNumArgs();++i) {
                if(name=="ncclCommInitRank"&&i==3)continue; // registered communicator rank projection
                observation(sid,call->getArg(i),"argument:"+std::to_string(i));
            }
            // Source range endpoints may come from CUDA's public API aliases;
            // the expression itself remains in the user's main file.
            auto range=call->getSourceRange();
            if(range.getEnd().isMacroID())fail(call,"function-like macros around CUDA calls require an ordinary source call");
            auto begin=sm.getExpansionLoc(range.getBegin()),finish=sm.getExpansionLoc(range.getEnd());
            const auto original=text(SourceRange(begin,finish));
            std::string code="([&]() { ::flexeva::code::Scope __fx_emit("+std::to_string(sid)+"ULL, FLEXEVA_CODE_EMISSION, "+values(call)+"); ";
            if(name=="ncclCommInitRank") {
                if(call->getNumArgs()!=4)fail(call,"unexpected ncclCommInitRank signature");
                const auto pointer=text(call->getArg(0)->getSourceRange());
                code+="auto __fx_result = "+original+"; if (__fx_result == 0) ::flexeva::code::communicator(reinterpret_cast<std::uint64_t>(* ("+pointer+")), 0, static_cast<std::uint32_t>("+text(call->getArg(3)->getSourceRange())+")); return __fx_result; }())";
            } else code+="return "+original+"; }())";
            replacements.emplace_back(offset(begin),end(finish),std::move(code));
            return true;
        }
        if(std::find(request.ground_functions.begin(),request.ground_functions.end(),qualified)!=request.ground_functions.end()) {
            if(!scalar(call->getType()))fail(call,"registered Ground functions must return a scalar");
            const auto sid=site(call,"ground",qualified);
            std::string code="::flexeva::code::ground<"+typeName(call->getType())+">("+std::to_string(sid)+"ULL, {";
            for(unsigned i=0;i<call->getNumArgs();++i) {
                if(!scalar(call->getArg(i)->getType())||call->getArg(i)->HasSideEffects(context))fail(call,"Ground inputs must be side-effect-free scalar expressions");
                if(i)code+=",";
                code+="::flexeva::code::scalar("+text(call->getArg(i)->getSourceRange())+")";
            }
            code+="})";replace(call->getSourceRange(),std::move(code));
            sharing_complete=false;share_reasons.push_back("Ground invocation requires a separately certified rank rule");
            return true;
        }
        if(auto* body=target->getDefinition();body&&local(body->getLocation())) {
            calls[current->getCanonicalDecl()].insert(body->getCanonicalDecl());
            for(auto* arg:call->arguments())if(tainted(arg)) {
                sharing_complete=false;share_reasons.push_back("rank-dependent helper call requires an interprocedural rank certificate");
            }
            const auto sid=site(call,"function",qualified);owned_call_sites[sid]=body->getCanonicalDecl();
            A writes;for(auto* var:effects(call))writes.push_back(std::to_string(variable(var)));
            (*sites.back().getAsObject())["defines"]=std::move(writes);
            (*sites.back().getAsObject())["semantic_digest"]=hash(semantic_text(call)+semantic_text(body->getBody()));
            replace(call->getSourceRange(),"([&]() -> decltype(auto) { ::flexeva::code::Scope __fx_call("+
                std::to_string(sid)+"ULL, FLEXEVA_CODE_FUNCTION, "+values(call)+"); return ("+text(call->getSourceRange())+"); }())");
            return true;
        }
        fail(call,"unregistered external call '"+qualified+"' has unknown effects");
    }
    bool VisitIfStmt(IfStmt* statement) {
        const auto sid=site(statement,"control","if");
        observation(sid,statement->getCond(),"predicate");
        controls.push_back(O{{"site",std::to_string(sid)},{"rank_dependent",tainted(statement->getCond())}});
        auto range=statement->getCond()->getSourceRange();
        // Inserts compose with an emission/Ground rewrite in the condition.
        insert(range.getBegin(),"::flexeva::code::branch("+std::to_string(sid)+"ULL, static_cast<bool>(");
        insert(Lexer::getLocForEndOfToken(range.getEnd(),0,sm,context.getLangOpts()),"))");
        return true;
    }
    bool VisitForStmt(ForStmt* statement) {
        const auto sid=site(statement,"control","for");
        if(statement->getCond())observation(sid,statement->getCond(),"loop");
        bracket(statement->getBody(),sid,"FLEXEVA_CODE_LOOP");return true;
    }
    bool VisitWhileStmt(WhileStmt* statement) {
        const auto sid=site(statement,"control","while");observation(sid,statement->getCond(),"loop");
        bracket(statement->getBody(),sid,"FLEXEVA_CODE_LOOP");return true;
    }
    bool VisitReturnStmt(ReturnStmt* statement) {
        bool nested_lambda=false;
        const Stmt* parent_statement=statement;
        while(parent_statement) {auto parents=context.getParents(*parent_statement);if(parents.empty())break;parent_statement=parents[0].get<Stmt>();if(isa_and_nonnull<LambdaExpr>(parent_statement)){nested_lambda=true;break;}}
        if(request.torch && current==entry && !nested_lambda) {
            if(const auto* result=statement->getRetValue()) {
                insert(result->getBeginLoc(),"(::flexeva::code::TorchReturnGuard{}, ");
                insert(Lexer::getLocForEndOfToken(result->getEndLoc(),0,sm,context.getLangOpts()),")");
            } else replace(statement->getSourceRange(),"return ::flexeva::code::torch_cleanup()");
        }
        if(statement->getRetValue()&&tainted(statement->getRetValue())) {
            const auto sid=site(statement,"control","return");observation(sid,statement->getRetValue(),"return");
        }
        return true;
    }
    bool VisitCXXForRangeStmt(CXXForRangeStmt* statement) {
        if(!request.torch)fail(statement,"range-for requires restricted Torch effects");
        const auto sid=site(statement,"control","range-for");observation(sid,statement->getRangeInit(),"range");bracket(statement->getBody(),sid,"FLEXEVA_CODE_LOOP");return true;
    }
    bool VisitStmt(Stmt* statement) {
        if(request.torch) {
            if(isa<SwitchStmt>(statement)||isa<DoStmt>(statement)||isa<ConditionalOperator>(statement)) {
                sharing_complete=false;share_reasons.push_back("conservative rank control: "+std::string(statement->getStmtClassName()));
            }
            if(isa<AsmStmt>(statement)||isa<IndirectGotoStmt>(statement))fail(statement,"inline assembly/indirect goto are outside a mutable training region");
            if(isa<LambdaExpr>(statement)||isa<CXXNewExpr>(statement)||isa<CXXDeleteExpr>(statement)||isa<GotoStmt>(statement)) {
                local_reuse_complete=false;sharing_complete=false;
            }
            return true;
        }
        if(isa<CXXThrowExpr>(statement)||isa<CXXTryStmt>(statement)||isa<GotoStmt>(statement)||
           isa<IndirectGotoStmt>(statement)||isa<AsmStmt>(statement)||isa<CXXNewExpr>(statement)||
           isa<CXXDeleteExpr>(statement)||isa<LambdaExpr>(statement)||isa<CXXForRangeStmt>(statement)||
           isa<SwitchStmt>(statement)||isa<DoStmt>(statement))
            fail(statement,"unsupported restricted-host control/effect syntax: "+std::string(statement->getStmtClassName()));
        return true;
    }
    bool VisitUnaryOperator(UnaryOperator* expr) {
        if(!request.torch && expr->getOpcode()==UO_Deref)fail(expr,"arbitrary pointer dereference is outside the host effect model");
        if(expr->isIncrementDecrementOp()) {
            const auto* ref=dyn_cast<DeclRefExpr>(expr->getSubExpr()->IgnoreParenImpCasts());
            const auto* var=ref?dyn_cast<VarDecl>(ref->getDecl()):nullptr;
            if(!var||var->hasGlobalStorage()||!scalar(var->getType())) {if(request.torch){local_reuse_complete=false;return true;}fail(expr,"increment requires a local scalar variable");}
            if(!inside_emission(expr))update(expr,var);
        }
        return true;
    }
    bool VisitDeclRefExpr(DeclRefExpr* expr) {
        if(request.torch)if(auto* var=dyn_cast<VarDecl>(expr->getDecl());var&&var->hasGlobalStorage()&&!var->getType().isConstQualified()) {
            local_reuse_complete=false;sharing_complete=false;
        }
        if(auto* var=dyn_cast<VarDecl>(expr->getDecl());var&&var->hasGlobalStorage()&&!var->getType().isConstQualified()&&!request.torch)
            fail(expr,"mutable global inputs/effects require an explicit frontend contract");
        return true;
    }
    bool VisitArraySubscriptExpr(ArraySubscriptExpr* expr){if(!request.torch)fail(expr,"host/device array reads are outside the scalar input contract");return true;}
    bool VisitMemberExpr(MemberExpr* expr){if(!request.torch)fail(expr,"record member reads require an explicit host effect contract");return true;}
    bool VisitCXXConstructExpr(CXXConstructExpr* expr) {
        const auto* constructor=expr->getConstructor();
        if(!request.torch && !constructor->isTrivial())fail(expr,"nontrivial host construction is outside the effect contract");
        if(request.torch && !constructor->isTrivial()) {
            const auto name=constructor->getQualifiedNameAsString();
            const bool known=local(constructor->getLocation())||name.starts_with("at::")||name.starts_with("torch::")||name.starts_with("c10::")||name.starts_with("std::");
            if(!known){local_reuse_complete=false;sharing_complete=false;share_reasons.push_back("conservative constructor effect: "+name);}
            if(name.starts_with("std::thread::")||name.starts_with("std::jthread::"))fail(expr,"host threads are outside the synchronous training-window contract");
        }
        return true;
    }
    bool VisitBinaryOperator(BinaryOperator* expr) {
        if(expr->isAssignmentOp()) {
            auto* ref=dyn_cast<DeclRefExpr>(expr->getLHS()->IgnoreParenImpCasts());
            auto* var=ref?dyn_cast<VarDecl>(ref->getDecl()):nullptr;
            if(!var||var->hasGlobalStorage()){if(request.torch){local_reuse_complete=false;return true;}fail(expr,"only local direct assignments have a complete host effect contract");}
            if(expr->getOpcode()==BO_Assign&&scalar(var->getType())) {
                auto sid=site(expr,"definition",var->getNameAsString());
                (*sites.back().getAsObject())["variable"]=std::to_string(variable(var));
                auto range=expr->getRHS()->getSourceRange();
                insert(range.getBegin(),"::flexeva::code::define("+std::to_string(sid)+"ULL, static_cast<"+var->getType().getUnqualifiedType().getAsString()+">(");
                insert(Lexer::getLocForEndOfToken(range.getEnd(),0,sm,context.getLangOpts()),"))");
            } else if(scalar(var->getType())&&!inside_emission(expr))update(expr,var);
        }
        return true;
    }
    bool inside_emission(const Stmt* statement) const {
        const auto* next=statement;
        while(next){auto parents=context.getParents(*next);if(parents.empty())break;next=parents[0].get<Stmt>();
            if(const auto* call=dyn_cast_or_null<CallExpr>(next))if(const auto* target=call->getDirectCallee();target&&emission(target->getName()))return true;}
        return false;
    }
    void update(Expr* expression,const VarDecl* variable_decl) {
        const auto sid=site(expression,"definition",variable_decl->getNameAsString());
        (*sites.back().getAsObject())["variable"]=std::to_string(variable(variable_decl));
        replace(expression->getSourceRange(),"::flexeva::code::update("+std::to_string(sid)+"ULL, "+variable_decl->getNameAsString()+
            ", [&]() -> decltype(auto) { return ("+text(expression->getSourceRange())+"); })");
    }
    bool VisitVarDecl(VarDecl* var) {
        if(var->getType().isVolatileQualified()||(!request.torch&&var->getType()->isReferenceType()))
            fail(var->getInit(),"volatile variables and reference aliases require an explicit host effect contract");
        if(var->isStaticLocal())fail(var->getInit(),"static mutable host state is outside the frontend contract");
        if(const auto* record=var->getType()->getAsCXXRecordDecl();!request.torch&&record&&!record->hasTrivialDestructor())
            fail(var->getInit(),"nontrivial host destruction is outside the effect contract");
        if(request.torch && var->hasInit() && !scalar(var->getType()) && !var->isImplicit()) {
            auto parents=context.getParents(*var);
            const auto* declaration=parents.empty()?nullptr:parents[0].get<DeclStmt>();
            auto contexts=context.getParents(*var);if(declaration)contexts=context.getParents(*declaration);
            if(declaration && !contexts.empty() && contexts[0].get<CompoundStmt>()) {
                const auto sid=site(var->getInit(),"definition",var->getNameAsString());
                (*sites.back().getAsObject())["variable"]=std::to_string(variable(var));
                if(isa<CXXConstructExpr>(var->getInit()->IgnoreImplicit()))(*sites.back().getAsObject())["semantic_digest"]=hash(semantic_text(var->getInit())+owned_class_semantics);
                std::string values=torch_values(var->getInit());values.pop_back();
                if(values.back()!='(')values+=",";values+=var->getNameAsString()+")";
                const auto after=Lexer::findLocationAfterToken(declaration->getEndLoc(),tok::semi,sm,context.getLangOpts(),false);
                if(after.isValid())insert(after," ::flexeva::code::torch_definition("+std::to_string(sid)+"ULL, "+values+"); ");
            }
        }
        if(var->hasInit()&&scalar(var->getType())&&!var->isImplicit()&&!var->getType()->isReferenceType()) {
            auto sid=site(var->getInit(),"definition",var->getNameAsString());
            (*sites.back().getAsObject())["variable"]=std::to_string(variable(var));
            auto range=var->getInit()->getSourceRange();
            insert(range.getBegin(),"::flexeva::code::define("+std::to_string(sid)+"ULL, static_cast<"+var->getType().getUnqualifiedType().getAsString()+">(");
            insert(Lexer::getLocForEndOfToken(range.getEnd(),0,sm,context.getLangOpts()),"))");
        }
        return true;
    }

    void cfg(FunctionDecl* function) {
        CFG::BuildOptions options;options.AddInitializers=true;
        auto graph=CFG::buildCFG(function,function->getBody(),&context,options);
        if(!graph)throw std::runtime_error("Clang failed to construct host CFG");
        using Def=std::pair<const VarDecl*,std::uint64_t>;
        using State=std::set<Def>;
        std::map<unsigned,State> incoming,outgoing;
        std::map<const Stmt*,std::uint64_t> nodes;
        std::map<unsigned,std::vector<const Stmt*>> statements;
        A blocks,edges;
        for(auto* block:*graph) {
            A node_ids,successors;
            unsigned position=0;
            for(const auto& element:*block)if(auto item=element.getAs<CFGStmt>()) {
                const auto* stmt=item->getStmt();
                auto number=id(function->getQualifiedNameAsString()+":cfg:"+std::to_string(block->getBlockID())+":"+std::to_string(position++));
                nodes[stmt]=number;statements[block->getBlockID()].push_back(stmt);
                node_ids.push_back(O{{"id",std::to_string(number)},{"syntax",stmt->getStmtClassName()},{"source",text(stmt->getSourceRange())}});
            }
            for(auto successor:block->succs())if(auto* next=successor.getReachableBlock())successors.push_back(next->getBlockID());
            blocks.push_back(O{{"id",block->getBlockID()},{"statements",std::move(node_ids)},{"successors",std::move(successors)}});
        }
        auto writes=[&](const Stmt* stmt) {
            std::set<const VarDecl*> out;
            if(auto* decl=dyn_cast<DeclStmt>(stmt))for(auto* d:decl->decls())if(auto* v=dyn_cast<VarDecl>(d))out.insert(v->getCanonicalDecl());
            const Expr* target=nullptr;
            if(auto* binary=dyn_cast<BinaryOperator>(stmt);binary&&binary->isAssignmentOp())target=binary->getLHS();
            if(auto* unary=dyn_cast<UnaryOperator>(stmt);unary&&unary->isIncrementDecrementOp())target=unary->getSubExpr();
            if(target)if(auto* ref=dyn_cast<DeclRefExpr>(target->IgnoreParenImpCasts()))if(auto* var=dyn_cast<VarDecl>(ref->getDecl()))out.insert(var->getCanonicalDecl());
            if(auto* call=dyn_cast<CallExpr>(stmt)){auto changes=effects(call);out.insert(changes.begin(),changes.end());}
            return out;
        };
        auto transfer=[&](unsigned block,State state) {
            for(auto* stmt:statements[block])for(auto* var:writes(stmt)) {
                for(auto it=state.begin();it!=state.end();)if(it->first==var)it=state.erase(it);else ++it;
                state.emplace(var,nodes[stmt]);
            }
            return state;
        };
        bool changed=true;
        while(changed) {
            changed=false;
            for(auto* block:*graph) {
                State merged;
                for(auto predecessor:block->preds())if(auto* previous=predecessor.getReachableBlock())
                    merged.insert(outgoing[previous->getBlockID()].begin(),outgoing[previous->getBlockID()].end());
                auto after=transfer(block->getBlockID(),merged);
                if(incoming[block->getBlockID()]!=merged||outgoing[block->getBlockID()]!=after){incoming[block->getBlockID()]=std::move(merged);outgoing[block->getBlockID()]=std::move(after);changed=true;}
            }
        }
        std::set<std::tuple<std::uint64_t,std::uint64_t,std::string>> unique;
        for(auto* block:*graph) {
            auto state=incoming[block->getBlockID()];
            for(auto* stmt:statements[block->getBlockID()]) {
                for(auto* var:refs(stmt))for(const auto& [defined,producer]:state)if(var==defined&&producer!=nodes[stmt])unique.emplace(producer,nodes[stmt],var->getNameAsString());
                for(auto* var:writes(stmt)) {for(auto it=state.begin();it!=state.end();)if(it->first==var)it=state.erase(it);else ++it;state.emplace(var,nodes[stmt]);}
            }
        }
        for(const auto& [producer,consumer,variable]:unique)edges.push_back(O{{"producer",std::to_string(producer)},{"consumer",std::to_string(consumer)},{"variable",variable}});
        cfgs.push_back(O{{"function",function->getQualifiedNameAsString()},{"entry",graph->getEntry().getBlockID()},
                         {"blocks",std::move(blocks)},{"def_use",std::move(edges)}});
    }

    void run() {
        for(auto* declaration:context.getTranslationUnitDecl()->decls())collect(declaration);
        for(auto* function:owned)if(function->getQualifiedNameAsString()==request.entry) {
            if(entry)throw std::runtime_error("ambiguous overloaded C++ entry; provide a unique entry wrapper");entry=function;
        }
        if(!entry)throw std::runtime_error("C++ entry definition not found: "+request.entry);
        if(entry->isVariadic()||(!request.extension&&!entry->getReturnType()->isVoidType()&&!scalar(entry->getReturnType())))
            fail(entry->getBody(),"C++ entry must return void or a scalar and cannot be variadic");
        bool has_rank=false,has_world=false;
        A parameters;
        for(auto* parameter:entry->parameters()) {
            auto name=parameter->getNameAsString();const auto type=request.extension&&!scalar(parameter->getType())?parameter->getType().getAsString():typeName(parameter->getType());
            if((name==request.rank_parameter||name==request.world_parameter)&&
               (!parameter->getType()->isIntegerType()||parameter->getType()->isBooleanType()))
                fail(entry->getBody(),"rank and world_size must be 32/64 bit integer parameters");
            if(name==request.rank_parameter){has_rank=true;rank_tainted.insert(parameter->getCanonicalDecl());}
            if(name==request.world_parameter)has_world=true;
            parameters.push_back(O{{"name",name},{"type",type},{"role",name==request.rank_parameter?"rank":name==request.world_parameter?"world_size":"argument"}});
        }
        if(!request.extension&&(!has_rank||!has_world))fail(entry->getBody(),"entry must expose the named rank and world_size scalar parameters");
        for(auto* function:owned)definitions.TraverseStmt(function->getBody());
        bool changed=true;
        while(changed){changed=false;for(const auto& [var,defs]:definitions.definitions)for(auto* expr:defs)if(tainted(expr))changed=rank_tainted.insert(var).second||changed;}
        // Analyze/instrument the reachable direct-call closure only. Unrelated
        // application entrypoints and helpers are not part of this invocation.
        std::set<const FunctionDecl*> visited;
        std::vector<FunctionDecl*> pending{entry};
        for(std::size_t i=0;i<pending.size();++i) {
            auto* function=pending[i];if(!visited.insert(function->getCanonicalDecl()).second)continue;
            if(function->isVariadic()||(!request.torch&&function->getReturnType()->isReferenceType()))
                fail(function->getBody(),"variadic helpers and reference results require an explicit host effect contract");
            for(auto* parameter:function->parameters())
                if(parameter->getType().isVolatileQualified()||(!request.torch&&parameter->getType()->isReferenceType()))
                    fail(function->getBody(),"volatile and reference parameters require an explicit host effect contract");
            current=function;cfg(function);
            const auto sid=site(function->getBody(),function==entry?"entry":"function",function->getQualifiedNameAsString());
            if(request.torch && function==entry) {
                auto* compound=dyn_cast<CompoundStmt>(function->getBody());if(!compound)fail(function->getBody(),"entry requires a function body");
                const auto cleanup_sid=site(function->getBody(),"cleanup","automatic-destruction");
                insert(compound->getRBracLoc()," ::flexeva::code::torch_cleanup(); ");
                std::string args;for(auto* parameter:function->parameters()){if(!args.empty())args+=",";args+=parameter->getNameAsString();}
                insert(Lexer::getLocForEndOfToken(compound->getLBracLoc(),0,sm,context.getLangOpts()),
                    " ::flexeva::code::TorchWindow __fx_window("+std::to_string(sid)+"ULL, "+std::to_string(cleanup_sid)+"ULL, ::flexeva::code::torch_values("+args+")); ");
            } else bracket(function->getBody(),sid,function==entry?"FLEXEVA_CODE_ENTRY":"FLEXEVA_CODE_FUNCTION");
            functions.push_back(O{{"name",function->getQualifiedNameAsString()},{"semantic_digest",hash(text(function->getSourceRange()))}});
            TraverseStmt(function->getBody());
            for(auto* callee:calls[function->getCanonicalDecl()]) {
                if(callee==entry->getCanonicalDecl())fail(function->getBody(),"recursive entry calls are unsupported");
                pending.push_back(const_cast<FunctionDecl*>(callee->getDefinition()));
            }
        }
        // Detect indirect recursion in the reachable direct-call graph.
        std::set<const FunctionDecl*> active,done;
        std::function<void(const FunctionDecl*)> walk=[&](const FunctionDecl* fn){if(done.contains(fn))return;if(!active.insert(fn).second)fail(fn->getBody(),"recursive helper calls are unsupported");for(auto* c:calls[fn])walk(c);active.erase(fn);done.insert(fn);};walk(entry->getCanonicalDecl());
        // Returning a value depends on the transitive source closure, including
        // helper -> helper calls whose caller text did not change.
        std::map<const FunctionDecl*,std::string> closures;
        std::function<std::string(const FunctionDecl*)> closure=[&](const FunctionDecl* function) {
            if(auto found=closures.find(function);found!=closures.end())return found->second;
            std::vector<std::pair<std::string,std::string>> callees;
            for(auto* callee:calls[function])callees.emplace_back(callee->getQualifiedNameAsString(),closure(callee));
            std::sort(callees.begin(),callees.end());
            auto semantic=semantic_text(function->getDefinition()->getBody());
            for(const auto& [name,digest]:callees)semantic+=name+":"+digest;
            return closures[function]=hash(semantic);
        };
        for(auto& value:sites) {
            auto* record=value.getAsObject();if(!record)continue;
            const auto key=std::stoull(record->getString("id")->str());
            if(auto found=owned_call_sites.find(key);found!=owned_call_sites.end())
                (*record)["semantic_digest"]=hash(record->getString("semantic_digest")->str()+closure(found->second));
        }
        A input_files,external_files;
        std::vector<std::pair<std::string,std::string>> fingerprints;
        for(auto it=sm.fileinfo_begin();it!=sm.fileinfo_end();++it) {
            const auto* file=it->first;
            bool invalid=false;auto buffer=sm.getBufferData(sm.translateFile(file),&invalid);
            if(!invalid)fingerprints.emplace_back(file->getName().str(),hash(buffer));
        }
        for(const auto& path:request.mutable_headers)fingerprints.emplace_back(path,hash(read(path)));
        for(const auto& path:request.extra_sources)fingerprints.emplace_back(path,hash(read(path)));
        std::sort(fingerprints.begin(),fingerprints.end());
        for(const auto& [path,digest]:fingerprints){input_files.push_back(O{{"path",path},{"sha256",digest}});if(path!=request.path&&std::find(request.mutable_headers.begin(),request.mutable_headers.end(),path)==request.mutable_headers.end()&&std::find(request.extra_sources.begin(),request.extra_sources.end(),path)==request.extra_sources.end())external_files.push_back(O{{"path",path},{"sha256",digest}});}
        A arguments;for(const auto& arg:request.arguments)arguments.push_back(arg);
        A links;for(const auto& arg:request.link_arguments)links.push_back(arg);
        const auto environment=encode(O{{"analyzer","clang-libtooling-14-v1"},{"compile_arguments",A(arguments)},
            {"link_arguments",A(links)},{"headers",std::move(external_files)}});
        A reasons;for(const auto& reason:share_reasons)reasons.push_back(reason);
        O plan{{"schema","flexeva-code-plan-v1"},{"language","cpp"},{"analyzer","clang-libtooling-14-v1"},
            {"entry",request.entry},{"parameters",std::move(parameters)},{"source_files",std::move(input_files)},
            {"compile_arguments",std::move(arguments)},{"functions",std::move(functions)},
            {"environment_identity",hash(environment)},{"link_arguments",std::move(links)},
            {"sites",std::move(sites)},{"cfg",std::move(cfgs)},{"controls",std::move(controls)},
            {"rank_observations",std::move(rank_observations)},{"rank_sharing_complete",sharing_complete},
            {"rank_sharing_obligations",std::move(reasons)},{"local_reuse_complete",local_reuse_complete},{"restricted_torch",request.torch},{"torch_allocator_policy",request.torch?"uncached-process-context-v1":"not-applicable"}};
        plan["source_digest"]=hash(encode(J(O(plan))));
        output.plan_json=encode(J(std::move(plan)));

        auto source=sm.getBufferData(sm.getMainFileID()).str();
        // Rewrite from original coordinates. Inserts may nest, but replacing
        // overlapping expressions is ambiguous and rejected explicitly.
        std::sort(replacements.begin(),replacements.end());
        for(std::size_t i=1;i<replacements.size();++i)if(std::get<0>(replacements[i])<std::get<1>(replacements[i-1]))
            fail(nullptr,"nested emission/Ground expression must be assigned to a local before use");
        std::map<unsigned,std::string> additions;
        for(auto& [where,code]:inserts)additions[where]+=code;
        std::map<unsigned,std::pair<unsigned,std::string>> edits;
        for(auto& [begin,finish,code]:replacements)edits.emplace(begin,std::make_pair(finish,std::move(code)));
        std::string transformed;
        for(unsigned i=0;i<=source.size();) {
            if(auto found=additions.find(i);found!=additions.end())transformed+=found->second;
            if(i==source.size())break;
            if(auto found=edits.find(i);found!=edits.end()){transformed+=found->second.second;i=found->second.first;}
            else transformed+=source[i++];
        }
        // Feed the admitted replacements through Clang's Rewriter. The one
        // resulting main-file buffer preserves the original include directives.
        rewrite.ReplaceText(sm.getLocForStartOfFile(sm.getMainFileID()),source.size(),transformed);
        std::string body;llvm::raw_string_ostream stream(body);rewrite.getEditBuffer(sm.getMainFileID()).write(stream);stream.flush();
        if(request.torch) {
            for(const auto& [from,to]:std::vector<std::pair<std::string,std::string>>{{"::flexeva::code::PathScope","::flexeva::code::TorchPathScope"},{"::flexeva::code::branch(","::flexeva::code::torch_branch("},{"::flexeva::code::define(","::flexeva::code::torch_define("},{"::flexeva::code::update(","::flexeva::code::torch_update("}}) {
                std::size_t position=0;while((position=body.find(from,position))!=std::string::npos){body.replace(position,from.size(),to);position+=to.size();}
            }
        }
        output.instrumented_source=std::string("#include <flexeva/code/")+(request.torch?"torch.hpp":"runtime.hpp")+">\n"+body+"\n";
        output.instrumented_source+="extern \"C\" const char* flexeva_code_plan_v1() noexcept { return R\"FXPLAN("+output.plan_json+")FXPLAN\"; }\n";
        if(request.torch)output.instrumented_source+="extern \"C\" int flexeva_code_prepare_v1() noexcept { try { ::flexeva::code::prepare_torch(); return 0; } catch (...) { return 1; } }\n";
        if(request.extension)return;
        output.instrumented_source+="extern \"C\" int flexeva_code_invoke_v1(const FlexevaCodeInvocationV1* inv, const FlexevaCodeCallbacksV1* callbacks, void* user) noexcept { return ::flexeva::code::invoke_guarded(inv, callbacks, user, [&] { ";
        for(auto* parameter:entry->parameters())if((parameter->getNameAsString()==request.rank_parameter||parameter->getNameAsString()==request.world_parameter)&&typeName(parameter->getType())=="std::int32_t")
            output.instrumented_source+="if (inv->"+std::string(parameter->getNameAsString()==request.rank_parameter?"rank":"world_size")+" > 2147483647u) throw std::runtime_error(\"rank/world_size exceeds the signed entry parameter range\"); ";
        const bool result=!entry->getReturnType()->isVoidType();
        if(result)output.instrumented_source+="if (!inv->result) throw std::runtime_error(\"C++ scalar entry requires a result slot\"); auto __fx_result = ";
        output.instrumented_source+=request.entry+"(";
        unsigned ordinal=0;
        for(unsigned i=0;i<entry->getNumParams();++i) {
            if(i)output.instrumented_source+=",";
            auto* parameter=entry->getParamDecl(i);auto name=parameter->getNameAsString();
            if(name==request.rank_parameter)output.instrumented_source+="static_cast<"+typeName(parameter->getType())+">(inv->rank)";
            else if(name==request.world_parameter)output.instrumented_source+="static_cast<"+typeName(parameter->getType())+">(inv->world_size)";
            else output.instrumented_source+="::flexeva::code::argument<"+typeName(parameter->getType())+">(*inv,"+std::to_string(ordinal++)+")";
        }
        output.instrumented_source+="); ";
        if(result)output.instrumented_source+="*inv->result = ::flexeva::code::scalar(__fx_result); ";
        output.instrumented_source+="}); }\n";
        const auto shim=output.instrumented_source.substr(output.instrumented_source.find("extern \"C\" int flexeva_code_invoke_v1"));
        auto v2=shim;auto change=[&](const std::string& from,const std::string& to){std::size_t position=0;while((position=v2.find(from,position))!=std::string::npos){v2.replace(position,from.size(),to);position+=to.size();}};
        change("flexeva_code_invoke_v1","flexeva_code_invoke_v2");change("FlexevaCodeCallbacksV1","FlexevaCodeCallbacksV2");change("invoke_guarded(","invoke_guarded_v2(");output.instrumented_source+=v2;

    }
    void collect(Decl* declaration) {
        if(auto* function=dyn_cast<FunctionDecl>(declaration)) {
            if(!sm.isInSystemHeader(function->getLocation())&&
               (function->hasAttr<ConstructorAttr>()||function->hasAttr<DestructorAttr>()))
                throw std::runtime_error("load/unload host hooks are outside the restricted frontend");
            if(function->isThisDeclarationADefinition()&&local(function->getLocation())&&(!isa<CXXMethodDecl>(function)||request.torch)&&!function->getLocation().isMacroID())owned.push_back(function);
        } else if(auto* record=dyn_cast<CXXRecordDecl>(declaration);record&&request.torch&&local(record->getLocation())) {if(record->isThisDeclarationADefinition()&&!record->getLocation().isMacroID())owned_class_semantics+=text(record->getSourceRange());for(auto* child:record->decls())collect(child);}
        else if(auto* templ=dyn_cast<FunctionTemplateDecl>(declaration);templ&&request.torch)collect(templ->getTemplatedDecl());
        else if(auto* ns=dyn_cast<NamespaceDecl>(declaration))for(auto* child:ns->decls())collect(child);
        else if(auto* linkage=dyn_cast<LinkageSpecDecl>(declaration))for(auto* child:linkage->decls())collect(child);
        else if(auto* var=dyn_cast<VarDecl>(declaration);var&&!request.torch&&!sm.isInSystemHeader(var->getLocation())) {
            const auto* record=var->getType()->getAsCXXRecordDecl();
            if((var->hasInit()&&!var->getInit()->isConstantInitializer(context,false))||
               (record&&!record->hasTrivialDestructor()))
                throw std::runtime_error("dynamic global initialization/destruction is outside the restricted host frontend: "+var->getNameAsString());
        }
    }
};

struct Consumer : ASTConsumer {
    const CppSource& request; CppAnalysis& output; std::string& error; Rewriter& rewrite;
    Consumer(const CppSource& r,CppAnalysis& o,std::string& e,Rewriter& w):request(r),output(o),error(e),rewrite(w){}
    void HandleTranslationUnit(ASTContext& context) override {
        if(context.getDiagnostics().hasErrorOccurred()){error="Clang rejected the input translation unit";return;}
        try {Analysis(context,rewrite,request,output).run();}catch(const std::exception& e){error=e.what();}
    }
};
struct Action : ASTFrontendAction {
    const CppSource& request;CppAnalysis& output;std::string& error;Rewriter rewrite;
    Action(const CppSource& r,CppAnalysis& o,std::string& e):request(r),output(o),error(e){}
    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance& compiler,llvm::StringRef) override {
        rewrite.setSourceMgr(compiler.getSourceManager(),compiler.getLangOpts());
        return std::make_unique<Consumer>(request,output,error,rewrite);
    }
};
} // namespace

CppAnalysis analyze_cpp(const CppSource& source) {
    if(source.path.empty()||source.entry.empty())throw std::runtime_error("C++ source and entry must be specified");
    CppAnalysis output;std::string error;auto resolved=source;
    if(!source.compilation_database.empty()) {
        auto database=tooling::CompilationDatabase::loadFromDirectory(source.compilation_database,error);
        if(!database)throw std::runtime_error("cannot load compilation database: "+error);
        error.clear(); // A successful JSON lookup may retain failed fixed-database diagnostics.
        const auto commands=database->getCompileCommands(source.path);
        if(commands.size()!=1)throw std::runtime_error("C++ source requires exactly one compile_commands entry");
        std::vector<std::string> flags{"-working-directory="+commands.front().Directory};
        const auto& command=commands.front().CommandLine;
        for(std::size_t i=1;i<command.size();++i) {
            if(command[i]=="-o"){++i;continue;}
            if(command[i]=="-c"||command[i]==source.path||command[i]==commands.front().Filename)continue;
            flags.push_back(command[i]);
        }
        flags.insert(flags.end(),source.arguments.begin(),source.arguments.end());resolved.arguments=std::move(flags);
    }
    auto arguments=resolved.arguments;
    if(std::none_of(arguments.begin(),arguments.end(),[](const auto& x){return x.starts_with("-std=");}))arguments.push_back("-std=c++20");
    for(const auto& arg:arguments)if(arg.starts_with("-std=")&&arg!="-std=c++20"&&arg!="-std=gnu++20")
        throw std::runtime_error("the typed C++ host frontend currently requires C++20 for analysis and compilation");
    arguments.push_back("-fno-delayed-template-parsing");
    // Analyze with precisely the injected runtime header environment, so its
    // macros/includes cannot change a conditional during subsequent compilation.
    arguments.push_back("-I" FLEXEVA_CODE_INCLUDE_DIR);
    arguments.push_back("-include");arguments.push_back(source.torch?"flexeva/code/torch.hpp":"flexeva/code/runtime.hpp");
    std::map<std::filesystem::path,std::string> owned_headers;
    for(const auto& path:source.mutable_headers) {
        const auto full=std::filesystem::weakly_canonical(path);owned_headers.emplace(full,read(full.string()));
        arguments.push_back("-I"+full.parent_path().string());
    }
    std::set<std::filesystem::path> included;
    std::function<std::string(const std::string&,const std::filesystem::path&)> expand;
    expand=[&](const std::string& content,const std::filesystem::path& path) {
        std::istringstream input(content);std::string result,line;unsigned lineno=0;
        const std::regex include(R"INCLUDE(^\s*#\s*include\s*["<]([^">]+)[">].*$)INCLUDE");
        while(std::getline(input,line)) {
            ++lineno;std::smatch match;
            if(std::regex_match(line,match,include)) {
                auto candidate=std::filesystem::weakly_canonical(path.parent_path()/match[1].str());
                auto found=owned_headers.find(candidate);
                if(found==owned_headers.end())for(auto item=owned_headers.begin();item!=owned_headers.end();++item)
                    if(item->first.filename()==match[1].str()){found=item;break;}
                if(found!=owned_headers.end()) {
                    if(included.insert(found->first).second)result+="\n#line 1 "+encode(J(found->first.string()))+"\n"+expand(found->second,found->first);
                    result+="\n#line "+std::to_string(lineno+1)+" "+encode(J(path.string()))+"\n";continue;
                }
            }
            result+=line+"\n";
        }
        return result;
    };
    auto content=expand(read(source.path),source.path);
    for(const auto& path:source.extra_sources) {
        arguments.push_back("-I"+std::filesystem::path(path).parent_path().string());
        content+="\n#line 1 "+encode(J(path))+"\n"+expand(read(path),path);
    }
    resolved.arguments=arguments;output.compile_arguments=arguments;
    if(!tooling::runToolOnCodeWithArgs(std::make_unique<Action>(resolved,output,error),content,arguments,source.path,"clang++-14")||!error.empty())
        throw std::runtime_error(error.empty()?"Clang LibTooling failed":error);
    return output;
}
} // namespace flexeva::code
