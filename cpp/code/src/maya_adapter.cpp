#include <flexeva/code/maya.hpp>

#include <llvm/Support/JSON.h>
#include <llvm/Support/raw_ostream.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <set>

namespace flexeva::code {
namespace {
namespace paper=maya::paper;
namespace rt=core::runtime;
using J=llvm::json::Value;
using O=llvm::json::Object;
using A=llvm::json::Array;
std::string encode(const J& value){std::string result;llvm::raw_string_ostream out(result);out<<value;return out.str();}
rt::Evidence bytes(const std::string& value){return {reinterpret_cast<const std::byte*>(value.data()),reinterpret_cast<const std::byte*>(value.data()+value.size())};}
template<class T> void append(rt::Evidence& out,const T& value){const auto* data=reinterpret_cast<const std::byte*>(&value);out.insert(out.end(),data,data+sizeof(T));}
void append(rt::Evidence& out,const std::string& value){append(out,std::uint64_t(value.size()));auto data=bytes(value);out.insert(out.end(),data.begin(),data.end());}
std::string string(const O& object,llvm::StringRef field){auto value=object.getString(field);if(!value)throw Error("invalid C++ CodePlan string: "+field.str());return value->str();}
std::uint64_t number(const O& object,llvm::StringRef field){return std::stoull(string(object,field));}
std::int64_t integer(const ScalarV1& value){
    detail::validate_scalar(value);
    switch(value.kind){
    case FLEXEVA_CODE_BOOL:return value.value.boolean;
    case FLEXEVA_CODE_I32:return value.value.i32;
    case FLEXEVA_CODE_U32:return value.value.u32;
    case FLEXEVA_CODE_I64:return value.value.i64;
    case FLEXEVA_CODE_U64:if(value.value.u64<=std::uint64_t(INT64_MAX))return value.value.u64;break;
    default:break;
    }
    throw Error("rank formula needs a signed-range integer parameter");
}
std::int64_t evaluate(const J& value,const std::map<std::string,std::int64_t>& parameters){
    const auto* object=value.getAsObject();if(!object)throw Error("invalid C++ rank expression");
    if(auto constant=object->getInteger("constant"))return *constant;
    if(auto parameter=object->getString("parameter")){auto found=parameters.find(parameter->str());if(found==parameters.end())throw Error("rank expression depends on unavailable input");return found->second;}
    auto op=object->getString("operator");auto* operands=object->getArray("operands");
    if(!op||!operands||operands->empty()||operands->size()>2)throw Error("unsupported rank formula");
    const auto left=evaluate((*operands)[0],parameters);
    if(operands->size()==1){if(*op=="!")return !left;if(*op=="~")return ~left;if(*op=="+")return left;if(*op=="-"&&left!=INT64_MIN)return -left;throw Error("invalid unary rank expression");}
    if(*op=="&&"&&!left)return 0;if(*op=="||"&&left)return 1;
    const auto right=evaluate((*operands)[1],parameters);
    if(*op=="<")return left<right;if(*op==">")return left>right;if(*op=="<=")return left<=right;if(*op==">=")return left>=right;
    if(*op=="==")return left==right;if(*op=="!=")return left!=right;if(*op=="&&")return left&&right;if(*op=="||")return left||right;
    if(*op=="&")return left&right;if(*op=="|")return left|right;if(*op=="^")return left^right;
    if((*op=="/"||*op=="%")&&right&&!(left==INT64_MIN&&right==-1))return *op=="/"?left/right:left%right;
    std::int64_t result=0;
    if(*op=="+"&&!__builtin_add_overflow(left,right,&result))return result;
    if(*op=="-"&&!__builtin_sub_overflow(left,right,&result))return result;
    if(*op=="*"&&!__builtin_mul_overflow(left,right,&result))return result;
    throw Error("overflow or unsupported C++ rank formula");
}
struct Site {
    std::string coordinate,semantic,kind;
    std::vector<std::uint64_t> uses,defines;
    std::uint64_t variable=0;
    bool state_effect=false;
};
struct Group {std::uint32_t representative;std::vector<std::uint32_t> members;std::string signature;};
struct Active {
    rt::Id id=0;std::vector<rt::Id> aliases,deps,facts;
    rt::Evidence packet;
    const Site* site=nullptr;
};
}

struct CppWorkload::Impl {
    Module module;
    J plan;
    std::map<std::uint64_t,Site> sites;
    bool can_share=false,has_ground=false,local_complete=false,torch=false;
    explicit Impl(const std::string& path):module(path),plan(nullptr){
        auto parsed=llvm::json::parse(module.plan_json());
        if(!parsed)throw Error("C++ module contains malformed CodePlan JSON");plan=std::move(*parsed);
        const auto* object=plan.getAsObject();
        if(!object||string(*object,"schema")!="flexeva-code-plan-v1"||string(*object,"language")!="cpp")throw Error("C++ module CodePlan version mismatch");
        auto* list=object->getArray("sites");if(!list)throw Error("C++ plan is missing source sites");
        for(const auto& item:*list){const auto* site=item.getAsObject();if(!site)throw Error("invalid C++ source site");
            Site info{string(*site,"coordinate"),string(*site,"semantic_digest"),string(*site,"kind"),{},{},0};
            auto ids=[&](llvm::StringRef key,std::vector<std::uint64_t>& into){if(auto* values=site->getArray(key))for(const auto& v:*values){auto text=v.getAsString();if(!text)throw Error("invalid C++ variable identity");into.push_back(std::stoull(text->str()));}};
            ids("uses",info.uses);ids("defines",info.defines);if(site->getString("variable"))info.variable=number(*site,"variable");
            info.state_effect=site->getBoolean("state_effect").getValueOr(false);
            has_ground=has_ground||info.kind=="ground";
            if(!sites.emplace(number(*site,"id"),std::move(info)).second)throw Error("C++ local site digest collision");
        }
        can_share=object->getBoolean("rank_sharing_complete").getValueOr(false);
        local_complete=object->getBoolean("local_reuse_complete").getValueOr(false);
        torch=object->getBoolean("restricted_torch").getValueOr(false);
    }
    std::vector<Group> groups(const CppInvocation& input)const{
        auto ranks=input.ranks;if(ranks.empty()){ranks.resize(input.world_size);std::iota(ranks.begin(),ranks.end(),0);}
        if(!input.world_size||ranks.empty())throw Error("C++ invocation needs a nonempty rank world");
        std::sort(ranks.begin(),ranks.end());if(std::adjacent_find(ranks.begin(),ranks.end())!=ranks.end()||ranks.back()>=input.world_size)throw Error("C++ logical ranks must be distinct and in range");
        for(const auto& value:input.arguments)detail::validate_scalar(value);
        const std::map<std::string,std::uint32_t> kinds{{"bool",FLEXEVA_CODE_BOOL},{"std::int32_t",FLEXEVA_CODE_I32},
            {"std::uint32_t",FLEXEVA_CODE_U32},{"std::int64_t",FLEXEVA_CODE_I64},{"std::uint64_t",FLEXEVA_CODE_U64},
            {"float",FLEXEVA_CODE_F32},{"double",FLEXEVA_CODE_F64}};
        std::size_t supplied=0;
        for(const auto& item:*plan.getAsObject()->getArray("parameters")){
            const auto& parameter=*item.getAsObject();const auto role=string(parameter,"role"),type=string(parameter,"type");
            if(role=="argument"){
                if(supplied>=input.arguments.size()||input.arguments[supplied++].kind!=kinds.at(type))throw Error("C++ entry argument count or scalar type mismatch");
            }else if(type=="std::int32_t"&&(role=="rank"?ranks.back():input.world_size)>std::uint32_t(INT32_MAX))
                throw Error("C++ rank/world_size exceeds the signed entry parameter range");
        }
        if(supplied!=input.arguments.size())throw Error("C++ entry argument count mismatch");
        std::map<std::string,std::vector<std::uint32_t>> buckets;
        for(auto rank:ranks){std::string signature="rank:"+std::to_string(rank);
            if(input.share_ranks&&can_share)try{
                std::map<std::string,std::int64_t> parameters;std::size_t index=0;
                for(const auto& item:*plan.getAsObject()->getArray("parameters")){
                    const auto& parameter=*item.getAsObject();auto role=string(parameter,"role");
                    if(role=="rank")parameters[string(parameter,"name")]=rank;
                    else if(role=="world_size")parameters[string(parameter,"name")]=input.world_size;
                    else {if(index>=input.arguments.size())throw Error("missing C++ entry argument");try{parameters[string(parameter,"name")]=integer(input.arguments[index]);}catch(const Error&){}++index;}
                }
                if(index!=input.arguments.size())throw Error("C++ entry argument count mismatch");
                signature="class";
                for(const auto& item:*plan.getAsObject()->getArray("rank_observations")){
                    const auto& observation=*item.getAsObject();signature+=":"+string(observation,"site")+":"+std::to_string(evaluate(*observation.get("expression"),parameters));
                }
            }catch(const Error&){signature="rank:"+std::to_string(rank);}
            buckets[signature].push_back(rank);
        }
        std::vector<Group> out;for(auto& [signature,members]:buckets)out.push_back({members.front(),std::move(members),signature});
        std::sort(out.begin(),out.end(),[](const auto& a,const auto& b){return a.representative<b.representative;});return out;
    }
};

namespace {
struct Execution {
    std::shared_ptr<CppWorkload::Impl> workload;
    CppInvocation invocation;
    std::vector<Group> groups;
    std::shared_ptr<CppExecutionStatistics> statistics;
    paper::Capture* capture=nullptr;
    const Group* group=nullptr;
    std::vector<Active> stack;
    std::map<std::uint64_t,rt::Id> writers;
    std::map<std::uint64_t,bool> predicates;
    std::map<std::uint64_t,rt::Id> source_sites;
    rt::Id last_ground=0;
    rt::Id last_value_producer=0,last_state_producer=0;
    std::string failure;

    const Site& site(std::uint64_t key){auto found=workload->sites.find(key);if(found==workload->sites.end())throw Error("native Code occurrence refers to an unknown source site");return found->second;}
    rt::Evidence coordinate(const OccurrenceV1& occurrence,std::uint32_t rank){
        rt::Evidence out;append(out,std::string("cpp-code-occurrence-v1"));append(out,rank);
        for(std::size_t i=0;i<occurrence.path_size;++i){append(out,site(occurrence.path[i].site).coordinate);append(out,occurrence.path[i].ordinal);append(out,occurrence.path[i].kind);}
        return out;
    }
    Active open(const OccurrenceV1& occurrence,const ScalarV1* inputs,std::size_t size,bool produces_trace){
        const auto& info=site(occurrence.site);Active active;active.site=&info;
        auto make=[&](std::uint32_t rank){auto key=coordinate(occurrence,rank);return capture->identity(0x435050434f444501ULL,key);};
        active.id=make(group->representative);
        rt::Evidence semantic=bytes(info.semantic);append(semantic,string(*workload->plan.getAsObject(),"environment_identity"));append(semantic,invocation.world_size);
        if(!workload->torch)for(const auto& value:invocation.arguments)append(semantic,value);
        for(std::size_t i=0;i<size;++i)append(semantic,inputs[i]);
        for(const auto& [branch,taken]:predicates){append(semantic,branch);append(semantic,taken);}
        for(auto variable:info.uses)if(auto found=writers.find(variable);found!=writers.end())active.deps.push_back(found->second);
        if(last_ground)active.deps.push_back(last_ground);
        if(workload->torch&&last_state_producer&&info.kind=="emission")active.deps.push_back(last_state_producer);
        if(last_value_producer){active.deps.push_back(last_value_producer);last_value_producer=0;}
        if(!stack.empty()&&(!workload->torch||stack.back().site->kind!="entry"))active.deps.push_back(stack.back().id);
        std::sort(active.deps.begin(),active.deps.end());active.deps.erase(std::unique(active.deps.begin(),active.deps.end()),active.deps.end());
        for(auto rank:group->members){paper::Region region;region.id=make(rank);region.rank=rank;region.semantic=semantic;
            region.semantic_complete=workload->local_complete&&info.kind!="cleanup";region.reuse_allowed=workload->local_complete&&info.kind!="cleanup";region.produces_trace=produces_trace;
            region.site_id=source_sites.at(occurrence.site);
            region.site_identity=bytes(info.coordinate);
            region.occurrence_identity=coordinate(occurrence,0);
            if(rank!=group->representative)region.execution_region=active.id;
            capture->declare_region(std::move(region),active.deps);active.aliases.push_back(make(rank));
        }
        active.packet=coordinate(occurrence,occurrence.rank);append(active.packet,occurrence.order);append(active.packet,occurrence.kind);
        append(active.packet,group->signature);append(active.packet,semantic.size());active.packet.insert(active.packet.end(),semantic.begin(),semantic.end());
        capture->begin_region(active.id);++statistics->occurrences;return active;
    }
    void close(Active active){
        capture->code_occurrence(active.id,active.packet,active.facts,active.deps);
        capture->end_region();
        for(auto var:active.site->defines)writers[var]=active.id;
        if(active.site->variable)writers[active.site->variable]=active.id;
        if(active.site->kind=="function")last_value_producer=active.id;
        if(workload->torch&&active.site->state_effect)last_state_producer=active.id;
    }
    template<class Function> int protect(Function&& fn)noexcept{try{fn();return 0;}catch(const std::exception& e){if(failure.empty())failure=e.what();return -1;}catch(...){if(failure.empty())failure="unknown native Code callback failure";return -1;}}
    static int begin(void* user,const OccurrenceV1* occurrence,const ScalarV1* values,std::size_t size,std::uint64_t* token){auto& self=*static_cast<Execution*>(user);return self.protect([&]{auto active=self.open(*occurrence,values,size,occurrence->kind==FLEXEVA_CODE_EMISSION||(self.workload->torch&&occurrence->kind==FLEXEVA_CODE_ENTRY));*token=active.id;self.stack.push_back(std::move(active));});}
    static int end(void* user,std::uint64_t token){auto& self=*static_cast<Execution*>(user);return self.protect([&]{if(self.stack.empty()||self.stack.back().id!=token)throw Error("C++ emission scopes are unbalanced");auto active=std::move(self.stack.back());self.stack.pop_back();self.close(std::move(active));});}
    static int branch(void* user,std::uint64_t site_id,int taken){auto& self=*static_cast<Execution*>(user);return self.protect([&]{self.predicates[site_id]=taken!=0;const auto value=scalar(taken!=0);self.close(self.open(current_occurrence(),&value,1,false));});}
    static int definition(void* user,std::uint64_t,ScalarV1 value){auto& self=*static_cast<Execution*>(user);return self.protect([&]{self.close(self.open(current_occurrence(),&value,1,false));});}
    static int ground(void* user,std::uint64_t site_id,const ScalarV1* values,std::size_t size,ScalarV1* result){auto& self=*static_cast<Execution*>(user);return self.protect([&]{
        auto active=self.open(current_occurrence(),values,size,false);rt::GroundRequest request;
        request.point_id=self.capture->identity(0x43505047524e4401ULL,bytes(self.site(site_id).coordinate));request.request_id=active.id;
        request.requested_lanes.assign(self.group->members.begin(),self.group->members.end());request.query=bytes(self.site(site_id).coordinate);
        const auto raw=std::as_bytes(std::span(values,size));request.inputs.semantic.assign(raw.begin(),raw.end());
        const auto resolution=self.capture->ground(std::move(request),raw,cpp_ground_inputs_schema);
        if(!resolution.value.payload||resolution.value.payload.layout().schema!=cpp_ground_result_schema||resolution.value.payload.size()!=sizeof(ScalarV1))throw Error("C++ Ground handler must return one typed ScalarV1");
        auto value=resolution.value.payload.bytes();std::memcpy(result,value.span().data(),sizeof(*result));detail::validate_scalar(*result);
        active.facts.push_back(resolution.fact_id);append(active.packet,*result);const auto producer=active.id;self.close(std::move(active));self.last_ground=producer;++self.statistics->ground_calls;
    });}
    static int communicator(void* user,std::uint64_t handle,std::uint64_t membership,std::uint32_t local){auto& self=*static_cast<Execution*>(user);return self.protect([&]{self.capture->communicator(handle,membership?membership:1,local);});}
    static int fail(void* user,const char* message){auto& self=*static_cast<Execution*>(user);if(self.failure.empty())self.failure=message?message:"C++ workload failed";return 0;}
    void run(paper::Capture& sink){
        capture=&sink;
        const CallbacksV1 callbacks{FLEXEVA_CODE_ABI_VERSION,0,begin,end,branch,definition,ground,communicator,fail};
        for(const auto& [key,info]:workload->sites)source_sites.emplace(key,capture->identity(0x4350505349544501ULL,bytes(info.coordinate)));
        for(const auto& selected:groups){group=&selected;writers.clear();predicates.clear();failure.clear();last_ground=0;last_value_producer=0;last_state_producer=0;ScalarV1 result{};
        if(workload->torch) {
            paper::Region preparation;
            const auto site_identity=bytes("cpp-torch-runtime-preparation-v1");
            auto identity=site_identity;append(identity,selected.representative);
            preparation.id=capture->identity(0x4350505052455001ULL,identity);
            preparation.rank=selected.representative;
            preparation.site_id=capture->identity(0x4350505052534901ULL,site_identity);preparation.site_identity=site_identity;preparation.occurrence_identity=site_identity;
            preparation.semantic=identity;preparation.semantic_complete=false;preparation.reuse_allowed=false;preparation.produces_trace=false;
            capture->declare_region(std::move(preparation),{});capture->begin_region(capture->identity(0x4350505052455001ULL,identity));
            try {workload->module.prepare();capture->end_region();}catch(...){try{capture->end_region();}catch(...){}throw;}
        }
            const InvocationV1 request{FLEXEVA_CODE_ABI_VERSION,selected.representative,invocation.world_size,0,invocation.arguments.data(),invocation.arguments.size(),&result};
            ++statistics->invocations;statistics->representatives.push_back(selected.representative);
            if(workload->module.invoke(request,callbacks,this)!=0)throw Error(failure.empty()?"C++ workload execution failed":failure);
            if(!stack.empty())throw Error("C++ workload left open output scopes");statistics->returns.push_back(result);
        }
        capture=nullptr;
    }
};
}

CppWorkload::CppWorkload(const std::string& library):impl_(std::make_shared<Impl>(library)){}
const std::string& CppWorkload::plan_json()const noexcept{return impl_->module.plan_json();}
paper::Input CppWorkload::input(const CppInvocation& invocation,std::shared_ptr<CppExecutionStatistics> statistics)const{
    auto execution=std::make_shared<Execution>();execution->workload=impl_;execution->invocation=invocation;
    execution->groups=impl_->groups(invocation);execution->statistics=statistics?std::move(statistics):std::make_shared<CppExecutionStatistics>();
    paper::Input out;out.dynamic_regions=true;out.frontend_evidence=bytes(plan_json());
    out.configuration=bytes(impl_->torch?"cpp-maya-frontend-v1:uncached-process-context-v1":"cpp-maya-frontend-v1");out.window_semantic=bytes(plan_json());append(out.window_semantic,invocation.world_size);
    for(const auto& value:invocation.arguments)append(out.window_semantic,value);
    for(const auto& group:execution->groups){append(out.window_semantic,group.representative);for(auto member:group.members)append(out.window_semantic,member);}
    out.window_complete=!impl_->has_ground&&impl_->local_complete;
    paper::Communicator comm;comm.id=1;comm.members.resize(invocation.world_size);std::iota(comm.members.begin(),comm.members.end(),0);out.communicators.push_back(std::move(comm));
    for(const auto& group:execution->groups)for(auto member:group.members)out.rank_projections.push_back({member,group.representative,{{1,1}}});
    out.execute=[execution](paper::Capture& capture){execution->run(capture);};return out;
}
} // namespace flexeva::code
