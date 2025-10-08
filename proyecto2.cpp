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

/* utilidades LCG para permutar (bijección):
   x' = (a*x + b) mod M, con gcd(a,M)=1  */
unsigned long long gcd_ull(unsigned long long a, unsigned long long b){
    while (b){ auto t=a%b; a=b; b=t; }
    return a;
}
unsigned long long choose_coprime(unsigned long long M){
    // elige un 'a' impar y coprimo a M
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
    // bijección dentro del intervalo [L0,R0)
    unsigned long long span = R0 - L0;
    unsigned long long y = ( (__int128)a * (x - L0) + b ) % span;
    return L0 + y;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ---------- parámetros (root parsea) ----------
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

    // ---------- Bcast ----------
    MPI_Bcast(&L,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&check_every,1,MPI_UNSIGNED_LONG_LONG,0,MPI_COMM_WORLD);
    MPI_Bcast(&batch_size,1,MPI_UNSIGNED_LONG_LONG,0,MPI_COMM_WORLD);
    int wm = (work_mode=="stride")?1:0;
    MPI_Bcast(&wm,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&permute,1,MPI_INT,0,MPI_COMM_WORLD);

    const int MAXS=256;
    char pbuf[MAXS]={0}, tbuf[MAXS]={0};
    if(rank==0){ strncpy(pbuf,prefix.c_str(),MAXS-1); strncpy(tbuf,target_suffix.c_str(),MAXS-1); }
    MPI_Bcast(pbuf,MAXS,MPI_CHAR,0,MPI_COMM_WORLD);
    MPI_Bcast(tbuf,MAXS,MPI_CHAR,0,MPI_COMM_WORLD);
    prefix=string(pbuf); target_suffix=string(tbuf);

    // ---------- tamaño del espacio ----------
    unsigned long long base = (unsigned long long)ALPH.size();
    unsigned long long N=1; for(int i=0;i<L;++i) N*=base;

    // ---------- parámetros de permutación ----------
    // a,b diferentes por rango o por rank para evitar correlación
    unsigned long long a_global = choose_coprime(N);
    unsigned long long b_global = (unsigned long long)(rank*1315423911u) % N;

    // ---------- división de trabajo ----------
    unsigned long long my_start=0,my_end=0,step=1;
    if (wm==0){ // block (como en diagrama, con Scatter)
        if(rank==0){
            vector<unsigned long long> starts(size), ends(size);
            unsigned long long q=N/size, r=N%size;
            for(int rk=0; rk<size; ++rk){
                unsigned long long s=(unsigned long long)rk*q + min<unsigned long long>(rk,r);
                unsigned long long e=s + q + (rk<r?1ULL:0ULL);
                starts[rk]=s; ends[rk]=e;
            }
            MPI_Scatter(starts.data(),1,MPI_UNSIGNED_LONG_LONG,&my_start,1,MPI_UNSIGNED_LONG_LONG,0,MPI_COMM_WORLD);
            MPI_Scatter(ends.data(),1,MPI_UNSIGNED_LONG_LONG,&my_end,1,MPI_UNSIGNED_LONG_LONG,0,MPI_COMM_WORLD);
        }else{
            MPI_Scatter(nullptr,1,MPI_UNSIGNED_LONG_LONG,&my_start,1,MPI_UNSIGNED_LONG_LONG,0,MPI_COMM_WORLD);
            MPI_Scatter(nullptr,1,MPI_UNSIGNED_LONG_LONG,&my_end,1,MPI_UNSIGNED_LONG_LONG,0,MPI_COMM_WORLD);
        }
        step = 1;
    }else{ // stride (intercalado, NO orden secuencial)
        my_start = rank;
        my_end   = N;
        step     = size;
    }

    // ---------- búsqueda ----------
    int local_found=0, global_found=0;
    string my_suffix;
    unsigned long long iter_since=0;

    if (wm==0){ // block + lotes
        unsigned long long i = my_start;
        while(i<my_end){
            unsigned long long lot_end = min(i + batch_size, my_end);
            while(i<lot_end){
                unsigned long long idx = i;
                if(permute){
                    // permuta dentro de [my_start, my_end)
                    unsigned long long a = choose_coprime(my_end - my_start);
                    unsigned long long b = (b_global + i) % (my_end - my_start);
                    idx = lcg_map_interval(i, my_start, my_end, a, b);
                }
                string suf = idx_to_suffix(idx, L);
                string full = prefix + suf;
                if (is_alive_sim(full, prefix, target_suffix)){
                    local_found=1; my_suffix=suf;
                    MPI_Allreduce(&local_found,&global_found,1,MPI_INT,MPI_LOR,MPI_COMM_WORLD);
                    break;
                }
                if(++iter_since>=check_every){
                    iter_since=0;
                    MPI_Allreduce(&local_found,&global_found,1,MPI_INT,MPI_LOR,MPI_COMM_WORLD);
                    if(global_found) break;
                }
                ++i;
            }
            if(global_found) break;
        }
    } else { // stride + lotes
        unsigned long long i = my_start;
        while(i<my_end){
            unsigned long long lot_last = i + batch_size*step; // último índice de este lote (no inclusivo)
            unsigned long long lot_end = min(lot_last, my_end);
            for (unsigned long long j=i; j<lot_end; j+=step){
                unsigned long long idx = j;
                if(permute){
                    idx = lcg_map_global(j, N, a_global, b_global);
                }
                string suf = idx_to_suffix(idx, L);
                string full = prefix + suf;
                if (is_alive_sim(full, prefix, target_suffix)){
                    local_found=1; my_suffix=suf;
                    MPI_Allreduce(&local_found,&global_found,1,MPI_INT,MPI_LOR,MPI_COMM_WORLD);
                    break;
                }
                if(++iter_since>=check_every){
                    iter_since=0;
                    MPI_Allreduce(&local_found,&global_found,1,MPI_INT,MPI_LOR,MPI_COMM_WORLD);
                    if(global_found) break;
                }
            }
            if(global_found) break;
            i = i + batch_size*step; // siguiente lote
        }
    }

    MPI_Allreduce(&local_found,&global_found,1,MPI_INT,MPI_LOR,MPI_COMM_WORLD);

    // ---------- Gather + Barrier + Fin ----------
    const int buf_len = L + 1;
    string sendbuf = local_found ? my_suffix : string();
    sendbuf.resize(buf_len,'\0');
    vector<char> recvbuf; if(rank==0) recvbuf.resize(buf_len*size);

    MPI_Gather(sendbuf.data(), buf_len, MPI_CHAR,
               rank==0 ? recvbuf.data() : nullptr, buf_len, MPI_CHAR,
               0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank==0){
        string found_suffix;
        for(int r=0;r<size;++r){
            const char* p=&recvbuf[r*buf_len];
            string s(p, strnlen(p, buf_len));
            if(!s.empty()){ found_suffix=s; break; }
        }
        if(global_found && !found_suffix.empty()) cout << prefix << found_suffix << "\n";
        else                                      cout << "NOT_FOUND\n";
    }

    MPI_Finalize();
    return 0;
}