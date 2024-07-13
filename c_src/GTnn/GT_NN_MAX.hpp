#include <immintrin.h>
#include <bits/stdc++.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;
using namespace std::chrono;


namespace GT{
    class GroupTestingNN{
        private:
            unsigned int dim, N, Nq;
            double** X;
            double** Q;
            double** min_index;
            double** max_index;

            unsigned int index_sz;
            string data_path, query_path, result_path;
            float rho;

            // analysis variable 
            vector<int> negPools;
            unsigned long int ntests;
            double agg_qtime = 0;
            double agg_gt_qtime = 0;
            double mean_precision = 0;
            double mean_recall = 0;

        public:
            const string algo_name = "GroupTestingSignedMax";
            
            vector<vector<int>> search_res; 
            vector<vector<int>> gt_res; 

            static string pathAppend(const string& p1, const string& p2);
            bool extractMatrix(string filename, double** X, bool transpose, int max_rows);
            void saveMatrix(string filename, double** mat, unsigned int r, unsigned int c);
            void recursive_mkdir(const char *dir);
            void createIndex(double** X_cum, double** X, int N);
            void precision_and_recall();
            void check_file(ofstream &file);
            void extractMinMax(int i);

            GroupTestingNN(string &data_path, string &query_path, string &result_path, string &dname, unsigned int N, unsigned int Nq, unsigned int dim, float rho);
            inline int find_index(int round, int branch);

            bool load_data();
            bool build_index();
            bool save_index(const string &filename);
            bool load_index(const string &filename);
            void search();
            void exhaustive_search();   
            void save_results();   
    };
}

string GT::GroupTestingNN::pathAppend(const string& p1, const string& p2) {

    char sep = '/';
    string tmp = p1;

#ifdef _WIN32
    sep = '\\';
#endif

    if (p1[p1.length()] != sep) 
    { 
        tmp += sep;
        return(tmp + p2);
    }
    else
        return(p1 + p2);
}
 
bool GT::GroupTestingNN::extractMatrix(string filename, double** X, bool transpose = false, int max_rows = -1)
{
    ifstream file;
    string line;
    int rnum = 0, cnum = 0;
    file.open(filename.c_str());
    if(file.fail())
    {
        cout << "Couldn't open file : " << filename << endl;
        return false;
    } 
    while (getline(file, line))
    {
        if(!transpose && max_rows > 0 && rnum == max_rows)
            break;
        istringstream level(line);

        string value;
        cnum = 0;
        while (getline(level, value, ','))
        {
            if(transpose && max_rows > 0 && cnum == max_rows)
                break;
            double num = stod(value);
            if(transpose)
                X[cnum][rnum] = num;
            else
                X[rnum][cnum] = num;

            cnum++;
        }
        rnum++;
    }
    file.close();

    return true;
}

 
void GT::GroupTestingNN::saveMatrix(string filename, double** mat, unsigned int r, unsigned int c)
{
    ofstream file;
    file.open(filename.c_str());
    if(file.fail())
    {
        cout << "Couldn't open file : " << filename << endl;
        exit(1);
    } 
    for(unsigned int i = 0; i < r; i++)
    {
        for(unsigned int j = 0; j < c; j++)
        {
            if(j == c-1)
                file << mat[i][j] << endl;
            else
                file << mat[i][j] << " , ";
        }
    }

    return;
}
 
void GT::GroupTestingNN::createIndex(double** X_cum, double** X, int N)
{
    return;
}


void GT::GroupTestingNN::precision_and_recall()
{
    mean_precision = 0;
    mean_recall = 0;

    for(unsigned int i=0; i<this->Nq; i++)
    {
        std::unordered_set<int> truthSet(this->gt_res[i].begin(), this->gt_res[i].end());
        int truePositives = 0;

        for (const auto& pred : this->search_res[i]) {
            if (truthSet.find(pred) != truthSet.end()) {
                truePositives++;
            }
        }
        mean_precision += this->search_res[i].size() > 0 ? (static_cast<double>(truePositives) / this->search_res[i].size()) : 1.0;
        mean_recall += this->gt_res[i].size() > 0 ? (static_cast<double>(truePositives) / this->gt_res[i].size()) : 1.0;

    }
    mean_precision /= this->Nq;
    mean_recall /= this->Nq;
    return ;
}
 
void GT::GroupTestingNN::check_file(ofstream &file)
{
    if(file.fail())
    {
        cout << "Unable to open file" << endl;
        exit(1);
    }
}
 
void GT::GroupTestingNN::recursive_mkdir(const char *dir) {
    char tmp[256];
    char *p = NULL;
    size_t len;

    snprintf(tmp, sizeof(tmp),"%s",dir);
    len = strlen(tmp);
    if (tmp[len - 1] == '/')
        tmp[len - 1] = 0;
    for (p = tmp + 1; *p; p++)
        if (*p == '/') {
            *p = 0;
            mkdir(tmp, S_IRWXU);
            *p = '/';
        }
    mkdir(tmp, S_IRWXU);
}


GT::GroupTestingNN::GroupTestingNN(string &data_path, string &query_path, string &result_path, string &dname, unsigned int N, unsigned int Nq, unsigned int dim, float rho)
{
    this->data_path = data_path;
    this->query_path = query_path;
    this->rho = rho;
    this->N = N;
    this->Nq = Nq;
    this->dim = dim;

    this->result_path = this->pathAppend(this->pathAppend(result_path, this->algo_name), dname + "_rho" + to_string(this->rho));
    struct stat st = {0};
    if (stat((this->result_path).c_str(), &st) == -1) {
        recursive_mkdir((this->result_path).c_str());
    }
}

bool GT::GroupTestingNN::load_data()
{
    this->X = new double *[this->N];
    for (unsigned int i = 0; i < this->N; i++)
    {
        X[i] = new double[this->dim];
        if (X[i] == nullptr)
        {
            cout << "Not enough memory" << endl;
            exit(1);
        }
    }

    this->Q = new double *[this->Nq];
    for (unsigned int i = 0; i < this->Nq; i++)
    {
        Q[i] = new double[this->dim];
        if (Q[i] == nullptr)
        {
            cout << "Not enough memory" << endl;
            exit(1);
        }
    }
    bool is_loaded = this->extractMatrix(this->data_path, this->X, false, this->N);
    is_loaded = is_loaded && this->extractMatrix(this->query_path, this->Q, false, this->Nq);

    return is_loaded;
}

bool GT::GroupTestingNN::build_index()
{
    vector<pair<int, int>> queue;
    queue.push_back(make_pair(0, this->N-1));
    int idx = 0;
    while (true)
    {
        int i = queue[idx].first;
        int j = queue[idx].second;
        int w = j - i + 1;
        if (w == 1) break;
        queue.push_back(make_pair(i,i+w/2-1));
        queue.push_back(make_pair(i+w/2,j));
        idx++;
    }
    this->index_sz = queue.size();

    min_index = new double *[index_sz];
    for (unsigned int i = 0; i < index_sz; i++)
    {
        min_index[i] = new double[this->dim];
        if (min_index[i] == nullptr)
        {
            cout << "Not enough memory" << endl;
            exit(1);
        }
    }
    max_index = new double *[index_sz];
    for (unsigned int i = 0; i < index_sz; i++)
    {
        max_index[i] = new double[this->dim];
        if (max_index[i] == nullptr)
        {
            cout << "Not enough memory" << endl;
            exit(1);
        }
    }

    for(int i=index_sz-1; i>=0; i--)
    {
        int si = queue[i].first;
        int ei = queue[i].second;

        if(2*i + 2 < (int)index_sz)
        {
            this->extractMinMax(i);
        }
        else
        {
            int w = ei - si + 1;
            if (w > 2)
            {
                throw std::runtime_error("Something is wrong with building index");
            } 
            for(unsigned int j=0; j<this->dim; j++)
            {
                this->min_index[i][j] = min(this->X[si][j], this->X[ei][j]);
                this->max_index[i][j] = max(this->X[si][j], this->X[ei][j]);
            }
        }
    }
    return true; 
}

bool GT::GroupTestingNN::save_index(const string &dirname)
{
    std::ofstream file_min(this->pathAppend(dirname, string("min_index.txt")));
    std::ofstream file_max(this->pathAppend(dirname, string("min_index.txt")));

    if (file_min.is_open() && file_max.is_open()) {
        for (unsigned int i = 0; i < this->index_sz; ++i) {
            for (unsigned int j = 0; j < this->dim; ++j) {
                file_min << this->min_index[i][j];
                file_max << this->max_index[i][j];

                if (j < (this->dim - 1)) 
                {
                    file_min << ",";
                    file_max << ",";
                }
            }
            file_min << "\n";
        }
        file_min.close();
        file_max.close();

        std::cout << "Matrix successfully saved to " << this->pathAppend(dirname, string("min_index.txt")) << std::endl;
        std::cout << "Matrix successfully saved to " << this->pathAppend(dirname, string("max_index.txt")) << std::endl;

    } else {
        std::cerr << "Unable to open file " << this->pathAppend(dirname, string("min_index.txt")) << " or " <<  this->pathAppend(dirname, string("max_index.txt"))<< std::endl;
    }
    return true;
}

bool GT::GroupTestingNN::load_index(const string &dirname)
{
    vector<pair<int, int>> queue;
    queue.push_back(make_pair(0, this->N-1));
    int idx = 0;
    while (true)
    {
        int i = queue[idx].first;
        int j = queue[idx].second;
        int w = j - i + 1;
        if (w == 1) break;
        queue.push_back(make_pair(i,i+w/2-1));
        queue.push_back(make_pair(i+w/2,j));
        idx++;
    }
    this->index_sz = queue.size();

    min_index = new double *[index_sz];
    for (unsigned int i = 0; i < index_sz; i++)
    {
        min_index[i] = new double[this->dim];
        if (min_index[i] == nullptr)
        {
            cout << "Not enough memory" << endl;
            exit(1);
        }
    }
    max_index = new double *[index_sz];
    for (unsigned int i = 0; i < index_sz; i++)
    {
        max_index[i] = new double[this->dim];
        if (max_index[i] == nullptr)
        {
            cout << "Not enough memory" << endl;
            exit(1);
        }
    }

    bool is_loaded = this->extractMatrix(this->pathAppend(dirname, "min_index.txt"), this->min_index, false, this->index_sz);
    is_loaded = is_loaded && this->extractMatrix(this->pathAppend(dirname, "max_index.txt"), this->max_index, false, this->index_sz);

    return is_loaded;
}

inline
int GT::GroupTestingNN::find_index(int round, int branch)
{
    int base = (1 << (round)) - 1;
    return base + branch;
} 

void GT::GroupTestingNN::search()
{
    this->search_res.resize(this->Nq, vector<int>());
    int mxl = log2(this->N) + 1, level, si, ei, nele, round, branch;
    double stk[5][mxl], sim;
    (this->negPools).resize(mxl, 0);
    this->ntests = 0;

    for (unsigned int i = 0; i < this->Nq; i++)
    {
        cout << "Query idx : " << i << endl;
        // Running binary splitting method for each query with t as threshold.
        level = 0, si = 0, ei = this->N - 1;
        auto start = high_resolution_clock::now();
        
        // Start
        sim = 0;
        for (unsigned int j = 0; j < this->dim; j++)
        {
            if (Q[i][j] > 0)
                sim += (max_index[0][j]) * Q[i][j];
            else 
                sim += (min_index[0][j]) * Q[i][j];
        }
        this->ntests++;

        stk[0][0] = si; // start index
        stk[1][0] = ei; // end index
        stk[2][0] = sim; // similarity
        stk[3][0] = 0; // round
        stk[4][0] = 0; // branch
        while (level >= 0)
        {
            si = stk[0][level];
            ei = stk[1][level];
            sim = stk[2][level];
            round = stk[3][level];
            branch = stk[4][level];

            if (sim >= this->rho)
            {
                nele = (ei - si + 1);

                if (nele == 1)
                {
                        this->search_res[i].push_back(si);
                        level -= 1;
                }
                else if (nele == 2)
                {
                    double rsim = 0, lsim = 0;
                    for (unsigned int j = 0; j < this->dim; j++)
                    {
                        lsim += X[si][j] * Q[i][j];
                    }
                    for (unsigned int j = 0; j < this->dim; j++)
                    {
                        rsim += X[ei][j] * Q[i][j];
                    }

                    if (lsim >= this->rho)
                        this->search_res[i].push_back(si);
                    else
                        negPools[round]++;
                    
                    if (rsim >= this->rho)
                        this->search_res[i].push_back(ei);
                    else
                        negPools[round]++;

                    level -= 1;
                    ntests += 2;
                }
                else
                {
                    int index = floor(nele / 2);
                    double rsim = 0, lsim = 0;
                    int left_index = find_index(round + 1, 2*branch);
                    int right_index = find_index(round + 1, 2*branch + 1);

                    for (unsigned int j = 0; j < this->dim; j++)
                    {
                        if (Q[i][j] > 0)
                            lsim += (max_index[left_index][j]) * Q[i][j];
                        else 
                            lsim += (min_index[left_index][j]) * Q[i][j];
                    }
                    for (unsigned int j = 0; j < this->dim; j++)
                    {
                        if (Q[i][j] > 0)
                            rsim += (max_index[right_index][j]) * Q[i][j];
                        else 
                            rsim += (min_index[right_index][j]) * Q[i][j];
                    }
                    ntests += 2;

                    stk[0][level] = si + index;
                    stk[1][level] = ei;
                    stk[2][level] = rsim;
                    stk[3][level] = round + 1;
                    stk[4][level] = 2*branch + 1;

                    level++;

                    stk[0][level] = si;
                    stk[1][level] = si + index - 1;
                    stk[2][level] = lsim;
                    stk[3][level] = round + 1;
                    stk[4][level] = 2*branch;
                }
            }
            else
            {
                this->negPools[round]++;
                level--;
            }
        }
        // End
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        this->agg_qtime += duration.count();
    }

}


void GT::GroupTestingNN::exhaustive_search()
{
    this->gt_res.resize(this->Nq, vector<int>());
    double sim = 0;
    for (unsigned int i = 0; i < this->Nq; i++)
    {
        cout << "Query idx : " << i << endl;
        auto start = high_resolution_clock::now();
        for (unsigned int row = 0; row <this->N; row++)
        {
            sim = 0;
            for(unsigned int e=0; e<this->dim; e++) sim += (this->X[row][e])*(this->Q[i][e]);
            if (sim >= this->rho)
            {
                this->gt_res[i].push_back(row);
            }
        }
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        agg_gt_qtime += duration.count();
    }
}

void GT::GroupTestingNN::extractMinMax(int i)
{
    int child1 = 2*i + 1;
    int child2 = 2*i + 2; 
    for(unsigned int j=0; j<this->dim; j++)
    {
        this->min_index[i][j] = min(this->min_index[child1][j], this->min_index[child2][j]);
        this->max_index[i][j] = max(this->max_index[child1][j], this->max_index[child2][j]);
    }
}



void GT::GroupTestingNN::save_results()
{
    precision_and_recall();
    ofstream fresults, faggregates, fgroundtruth;

    fresults.open(this->pathAppend(this->result_path, string("results.txt")));
    check_file(fresults);

    fgroundtruth.open(this->pathAppend(this->result_path, string("ground_truth.txt")));
    check_file(fgroundtruth);
    
    faggregates.open(this->pathAppend(this->result_path, string("agg.txt")));
    check_file(faggregates);

    for (unsigned int i = 0; i < this->Nq; i++)
    {   
        for(unsigned int p=0;  p < this->search_res[i].size(); p++)
        {
            if (p != this->search_res[i].size()-1) fresults << this->search_res[i][p] << " , " ;
            else fresults << this->search_res[i][p] << endl;
        }
    }

    for (unsigned int i = 0; i < this->Nq; i++)
    {   
        for(unsigned int p=0;  p < this->gt_res[i].size(); p++)
        {
            if (p != this->gt_res[i].size()-1) fgroundtruth << this->gt_res[i][p] << " , " ;
            else fgroundtruth << this->gt_res[i][p] << endl;
        }
    }

    // Dumping aggregates
    faggregates << "Algorithm : " << this->algo_name << endl;
    faggregates << "Avgerage pool search query time : " << this->agg_qtime/this->Nq << endl;
    faggregates << "Avgerage exhaustive search query time : " << this->agg_gt_qtime/this->Nq << endl;
    faggregates << "Average precision : " << this->mean_precision << endl;
    faggregates << "Average recall : " << this->mean_recall << endl;
    faggregates << "Number of dot products : " << this->ntests << endl;
    faggregates << "Number of neg pools : " ;
    for(unsigned int i = 0 ; i < this->negPools.size(); i++)
    {
        if (i != this->negPools.size()-1) faggregates << this->negPools[i] << " , " ;
        else faggregates << this->negPools[i] << endl;
    }
    
    // Safe close
    fresults.close();
    faggregates.close();
    fgroundtruth.close();
}  

