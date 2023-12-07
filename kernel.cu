template <typename T> __device__
inline T fwrap(const T x, const T len) {
    return x - len * floor(x / len);
}

template <typename T> __global__
void _wrap_particles(
    const int N,
    T* x, T* ux, T* uy, T* uz,
    const T* bulk_u, const T v_rms,
    const T* rand_vx, const T* rand_vy, const T* rand_vz,
    const T L_axial, const T dt
)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        x[i] = fma(ux[i], dt, x[i]);
        bool right = x[i] >= L_axial;
        bool left  = x[i] < 0;
        int sign = left ? -1 : 1;
        if (x[i] >= L_axial || x[i] < 0) {
            x[i] = fwrap<T>(x[i], L_axial);
            ux[i] = fma(v_rms, sign * abs(rand_vx[i]), bulk_u[0]);
            uy[i] = fma(v_rms, rand_vy[i], bulk_u[1]);
            uz[i] = fma(v_rms, rand_vz[i], bulk_u[2]);
        }
    }
}
