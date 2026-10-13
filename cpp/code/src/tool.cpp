#include <flexeva/code/frontend.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;
namespace {
void write(const fs::path& path,const std::string& text) {
    std::ofstream out(path,std::ios::binary);
    if(!out || !(out<<text))throw std::runtime_error("cannot write "+path.string());
}
void execute(std::vector<std::string> arguments) {
    std::vector<char*> argv;for(auto& value:arguments)argv.push_back(value.data());argv.push_back(nullptr);
    const auto child=fork();if(child<0)throw std::runtime_error("cannot fork compiler");
    if(!child){execvp(argv[0],argv.data());_exit(127);}
    int status=0;if(waitpid(child,&status,0)!=child||!WIFEXITED(status)||WEXITSTATUS(status))
        throw std::runtime_error("C++ instrumented shared-library compilation failed");
}
}
int main(int argc,char** argv) {
    try {
        flexeva::code::CppSource source;
        fs::path output;std::string compiler="clang++-14";bool build=false;
        std::vector<std::string> link_arguments;
        for(int i=1;i<argc;++i) {
            const std::string argument=argv[i];
            auto value=[&]{if(++i==argc)throw std::runtime_error("missing value for "+argument);return std::string(argv[i]);};
            if(argument=="--source")source.path=value();
            else if(argument=="--entry")source.entry=value();
            else if(argument=="--rank")source.rank_parameter=value();
            else if(argument=="--world-size")source.world_parameter=value();
            else if(argument=="--compile-commands")source.compilation_database=value();
            else if(argument=="--ground")source.ground_functions.push_back(value());
            else if(argument=="--output-dir")output=value();
            else if(argument=="--compiler")compiler=value();
            else if(argument=="--link-arg")link_arguments.push_back(value());
            else if(argument=="--build")build=true;
            else if(argument=="--torch")source.torch=true;
            else if(argument=="--mutable-header")source.mutable_headers.push_back(fs::absolute(value()).string());
            else if(argument=="--extra-source")source.extra_sources.push_back(fs::absolute(value()).string());
            else if(argument=="--extension"){source.extension=true;source.torch=true;}
            else if(argument=="--") {while(++i<argc)source.arguments.emplace_back(argv[i]);break;}
            else throw std::runtime_error("unknown option: "+argument);
        }
        if(output.empty()||source.path.empty()||source.entry.empty())
            throw std::runtime_error("usage: flexeva-code-cpp --source file.cpp --entry run --output-dir DIR [--build] [--ground function] [--link-arg ARG] -- COMPILE_FLAGS");
        source.path=fs::absolute(source.path).string();output=fs::absolute(output);
        if(fs::path(compiler).filename()!="clang++-14")throw std::runtime_error("C++ analysis and instrumented compilation require clang++-14");
        for(const auto& argument:link_arguments)if(!argument.starts_with("-l")&&!argument.starts_with("-L")&&
            !argument.starts_with("-Wl,-rpath,")&&!argument.ends_with(".so")&&!argument.ends_with(".a"))
                throw std::runtime_error("--link-arg only accepts -l/-L, runtime search paths and library files; put compilation flags after --");
        source.link_arguments=link_arguments;
        const auto result=flexeva::code::analyze_cpp(source);
        fs::create_directories(output);
        write(output/"plan.json",result.plan_json+"\n");
        write(output/"instrumented.cpp",result.instrumented_source);
        if(build) {
            std::vector<std::string> command{compiler,"-std=c++20","-shared","-fPIC","-O2","-I" FLEXEVA_CODE_INCLUDE_DIR,
                "-I"+fs::path(source.path).parent_path().string()};
            command.insert(command.end(),result.compile_arguments.begin(),result.compile_arguments.end());
            command.insert(command.end(),{(output/"instrumented.cpp").string(),"-L" FLEXEVA_CODE_RUNTIME_DIR,
                "-Wl,-rpath," FLEXEVA_CODE_RUNTIME_DIR,"-lflexeva_code_runtime"});
            command.insert(command.end(),link_arguments.begin(),link_arguments.end());
            command.insert(command.end(),{"-o",(output/"workload.so").string()});execute(std::move(command));
        }
        std::cout<<(output/"plan.json").string()<<'\n';
        return 0;
    } catch(const std::exception& e){std::cerr<<"flexeva-code-cpp: "<<e.what()<<'\n';return 1;}
}
