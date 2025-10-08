#include <mpi.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstring>

using namespace std;

static const string ALPH = "abcdefghijklmnopqrstuvwxyz0123456789";

string idx_to_suffix(unsigned long long idx, int L) {
    string s(L, ALPH[0]);
    const unsigned long long base = (unsigned long long)ALPH.size();
    for (int i = L - 1; i >= 0; --i) {
        s[i] = ALPH[idx % base];
        idx /= base;
    }
    return s;
}

bool is_alive_sim(const string& full, const string& target_prefix, const string& target_suffix) {
    if (full.size() != target_prefix.size() + target_suffix.size()) return false;
    return full.compare(target_prefix.size(), target_suffix.size(), target_suffix) == 0;
}

unsigned long long gcd_ull(unsigned long long a, unsigned long long b){
    while (b){ auto t=a%b; a=b; b=t; }
    return a;
}
unsigned long long choose_coprime(unsigned long long M){
    unsigned long long a = (M>3? M-1 : 1);
    if (a%2==0) --a;
    for (; a>=1; a-=2) if (gcd_ull(a,M)==1) return a;
    return 1;
}
inline unsigned long long lcg_map_global(unsigned long long x,
                                         unsigned long long M,
                                         unsigned long long a,
                                         unsigned long long b){
    return ( (__int128)a * x + b ) % M;
}
inline unsigned long long lcg_map_interval(unsigned long long x,
                                           unsigned long long L0,
                                           unsigned long long R0,
                                           unsigned long long a,
                                           unsigned long long b){
    unsigned long long span = R0 - L0;
    unsigned long long y = ( (__int128)a * (x - L0) + b ) % span;
    return L0 + y;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);

    // parámetros 
    string prefix = "host-A-";
    string target_suffix = "aaaaaaa";
    int L = 7;
    unsigned long long check_every = 50000;
    unsigned long long batch_size  = 1000000ULL;
    string work_mode = "block";   // block | stride
    int permute = 0;              // 0 = no, 1 = sí (no recorrer en orden)

    if (rank==0){
        for (int i=1;i<argc;++i){
            string a=argv[i]; auto eq=a.find('=');
            string k= eq==string::npos? a : a.substr(0,eq);
            string v= eq==string::npos? "" : a.substr(eq+1);
            if      (k=="--prefix") prefix=v;
            else if (k=="--target"){ target_suffix=v; L=(int)target_suffix.size(); }
            else if (k=="--len")    L=stoi(v);
            else if (k=="--check_every") check_every=stoull(v);
            else if (k=="--batch")       batch_size =stoull(v);
            else if (k=="--work")        work_mode  =v;     // block|stride
            else if (k=="--permute")     permute    =stoi(v); // 0|1
        }
        if ((int)target_suffix.size()!=L) target_suffix.resize(L,'a');
    }

}
