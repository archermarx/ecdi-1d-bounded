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
        x[i] = x[i] + ux[i] * dt;

        if (x[i] >= L_axial || x < 0) {
            x[i] = fmod(x[i], L_axial);
            ux[i] = v_rms * rand_vx[i] + bulk_u[0];
            uy[i] = v_rms * rand_vy[i] + bulk_u[1];
            uz[i] = v_rms * rand_vz[i] + bulk_u[2];
        }
    }
}