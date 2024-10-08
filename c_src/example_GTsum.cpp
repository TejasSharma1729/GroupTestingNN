#include <GT_NN_SUM.hpp>
#include <bits/stdc++.h>
#include <cstdlib>
#include <unistd.h>
using namespace GT;

using namespace std;
int main(int argc, char* argv[])
{
    std::string datasetName;
    std::string queryPath;
    std::string datasetPath;
    std::string resultPath;
    float rho = 0.8;
    unsigned int N = 1000000;
    unsigned int Nq = 10000;
    unsigned int dim = 1000;

    int opt;
    while ((opt = getopt(argc, argv, "n:q:d:r:R:N:Q:D:")) != -1) {
        switch (opt) {
            case 'n':
                datasetName = optarg;
                break;
            case 'q':
                queryPath = optarg;
                break;
            case 'd':
                datasetPath = optarg;
                break;
            case 'r':
                resultPath = optarg;
                break;
            case 'R':
                rho = std::stof(optarg);
                break;
            case 'N':
                N = std::stoul(optarg);
                break;
            case 'Q':
                Nq = std::stoul(optarg);
                break;
            case 'D':
                dim = std::stoul(optarg);
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " -d <dataset-name> -q <query-path> -p <dataset-path> -r <result-path> -R <rho> -N <N> -Q <Nq> -D <dim>\n";
                return EXIT_FAILURE;
        }
    }

    // Print the parsed values
    std::cout << "Dataset Name: " << datasetName << std::endl;
    std::cout << "Query Path: " << queryPath << std::endl;
    std::cout << "Dataset Path: " << datasetPath << std::endl;
    std::cout << "Result Path: " << resultPath << std::endl;
    std::cout << "Rho: " << rho << std::endl;
    std::cout << "N: " << N << std::endl;
    std::cout << "Nq: " << Nq << std::endl;
    std::cout << "Dim: " << dim << std::endl;
    GroupTestingNN search_engine(datasetPath, queryPath, resultPath, datasetName, N, Nq, dim, rho);
    
    search_engine.load_data();
    search_engine.build_index();
    
    // search_engine.save_index("<path>");
    // search_engine.load_index("<path>");
    // the search results are stored in search_engine.search_res : vector<vector<int>>
    search_engine.search();
    search_engine.exhaustive_search();   
    search_engine.save_results();


    cout << "Execution completed successfully" << endl;
    cout << "Script by Harsh-Sensei" << endl;
    return 0;
}
