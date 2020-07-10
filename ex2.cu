#include "ex2.h"
#include <cuda/atomic>
#include <stdio.h>
#include <vector>
#include <queue>
#define NSTREAMS 64
#define FREE -1
#define IMAGE_SIZE IMG_HEIGHT*IMG_WIDTH
#define REGS_PER_THREAD 32
#define NEEDED_SHARED 256*(sizeof(int) + sizeof(uchar)) +  sizeof(int) + 2*sizeof(uchar*);
#define N_IMG_PAIRS 10000
#define  QUEUE_SIZE 16 
#define  MIN_THREADS 256 

__device__ void prefix_sum(int arr[], int arr_size) {
    int tid = threadIdx.x;
    int increment;

    for (int stride = 1; stride < min(blockDim.x, arr_size); stride *= 2) {
        if (tid >= stride && tid < arr_size) {
            increment = arr[tid - stride];
        }
        __syncthreads();
        if (tid >= stride && tid < arr_size) {
            arr[tid] += increment;
        }
        __syncthreads();
    }
}

// Example single-threadblock kernel for processing a single image.
// Feel free to change it.
__device__ void process_image_kernel(uchar *in, uchar *out) {
    __shared__ int histogram[256];
    __shared__ uchar map[256];

    int tid = threadIdx.x;

    if (tid < 256) {
        histogram[tid] = 0;
    }
    __syncthreads();

    for (int i = tid; i < IMG_HEIGHT * IMG_HEIGHT; i += blockDim.x)
        atomicAdd(&histogram[in[i]], 1);

    __syncthreads();

    prefix_sum(histogram, 256);

    if (tid < 256) {
        float map_value = float(histogram[tid]) / (IMG_WIDTH * IMG_HEIGHT);
        map[tid] = ((uchar)(N_COLORS * map_value)) * (256 / N_COLORS);
    }

    __syncthreads();

    for (int i = tid; i < IMG_WIDTH * IMG_HEIGHT; i += blockDim.x) {
        out[i] = map[in[i]];
    }
}

__global__ void serial_process_image_kernel(uchar *in, uchar *out){
    process_image_kernel(in, out);
    return;
}

class streams_server : public image_processing_server{
private:
    // TODO define stream server context (memory buffers, streams, etc...)
    cudaStream_t streams[NSTREAMS];
    int stream_to_img[NSTREAMS];
    uchar *dimg_in;
    uchar *dimg_out;
    int last_img_id;

    int get_available_stream(){
        for(int i = 0 ; i < NSTREAMS ; i++){
            if(stream_to_img[i] == FREE){
                return i;
            }
        }
        return -1;
    }

public:
    streams_server(){
        // TODO initialize context (memory buffers, streams, etc...)
        for (int i=0 ; i<NSTREAMS ; i++){
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
            stream_to_img[i] = FREE;
        }
        CUDA_CHECK( cudaMalloc(&dimg_in, NSTREAMS * IMG_WIDTH * IMG_HEIGHT) );
        CUDA_CHECK( cudaMalloc(&dimg_out, NSTREAMS * IMG_WIDTH * IMG_HEIGHT) );
    }

    ~streams_server() override
    {
        /* free streams */
        for (int i=0 ; i<NSTREAMS ; i++)
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
        CUDA_CHECK( cudaFree(dimg_in) );
        CUDA_CHECK( cudaFree(dimg_out) );
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override{
        int free = this->get_available_stream();
        if(free == -1){
            return false;
        }
        int image_size = IMG_WIDTH * IMG_HEIGHT;
        stream_to_img[free] = img_id;
        CUDA_CHECK( cudaMemcpyAsync(&dimg_in[image_size  * free], img_in, image_size , cudaMemcpyHostToDevice, streams[free] ));
        serial_process_image_kernel<<<1, 1024,0, streams[free]>>>(&dimg_in[image_size  * free], &dimg_out[ image_size * free]);
        // process_image_kernel(&dimg_in[image_size  * free], &dimg_out[ image_size * free]);
        CUDA_CHECK( cudaMemcpyAsync(img_out, &dimg_out[image_size  * free], image_size , cudaMemcpyDeviceToHost, streams[free] ));
        return true;
    
    }

    bool dequeue(int *img_id) override{
        for(int i = 0 ; i< NSTREAMS ; i++){
            if( (cudaStreamQuery(streams[i]) == cudaSuccess) && (stream_to_img[i] != FREE)){
                *img_id = stream_to_img[i];
                stream_to_img[i] = FREE;
                return true;
            }
        }
        return false; 
    }
};
std::unique_ptr<image_processing_server> create_streams_server() {
    return std::make_unique<streams_server>();
}
//-----------------------------------------------------------------------------
//                              GENERAL STUFF START
//-----------------------------------------------------------------------------
int query_device(int thread_per_threadblock){
    //calculate optimal parameters per block
    int desired_regs = REGS_PER_THREAD * thread_per_threadblock;
    //calculate limitations
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int num_sm_properties = prop.multiProcessorCount;
    int threads_properties = prop.maxThreadsPerMultiProcessor;
    int registers_properties = prop.regsPerMultiprocessor;
    int shared_memory_properties = prop.sharedMemPerMultiprocessor;
    //calculate most strict limit of thread blocks per sm
    int res1 = threads_properties/thread_per_threadblock;
    int res2 = registers_properties/desired_regs;
    int res3 = shared_memory_properties/NEEDED_SHARED;
    int num_of_theadblock_res = min(res1, min(res2,res3 )) * num_sm_properties;
    return num_of_theadblock_res;
}

class producer_consumer_q {
    private:
    public:
        cuda::atomic<int> queue_tail;
        cuda::atomic<int> queue_head;
        uchar* images[QUEUE_SIZE];
        int image_index[QUEUE_SIZE];
        uchar* image_ptrs[QUEUE_SIZE];
        __device__ bool is_queue_empty(){
            return queue_tail.load(cuda::memory_order_acquire) == queue_head;
        }
        

        __device__ bool is_queue_full(){
            return queue_tail - queue_head.load(cuda::memory_order_acquire) == QUEUE_SIZE;
        }

        producer_consumer_q(){
            queue_head = 0;
            queue_tail = 0;
            for(int i = 0; i < QUEUE_SIZE; i++){
                image_ptrs[i] = NULL;
                images[i] = NULL;
                image_index[i] = FREE;
            }
        }
}PCQ;

__device__ bool enqueue_gpu(producer_consumer_q* q,int img_id) {
    int tail = q->queue_tail.load(cuda::memory_order_relaxed);
    while(q->is_queue_full()); // busy wait - no returning false
    int index = q->queue_tail % QUEUE_SIZE;
    q->image_index[index] = img_id;
    q->queue_tail.store((tail + 1), cuda::memory_order_release);
    return true;
}

__device__ bool dequeue_gpu(producer_consumer_q* q,int* img_id, uchar* &image, uchar* &current_image) {
    int head = q->queue_head.load(cuda::memory_order_relaxed);
    int index = q->queue_head % QUEUE_SIZE;
    if(q->is_queue_empty()){
        return false;
    }
    current_image = q->image_ptrs[index];
    image = q->images[index];
    *img_id = q->image_index[index];
    q->queue_head.store((head + 1), cuda::memory_order_release);
    return true;
}

__global__ void worker(producer_consumer_q *cpu_gpu_q, producer_consumer_q *gpu_cpu_q, bool *stop) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    __shared__ int img_id;
    __shared__ uchar* img_in;
    __shared__ uchar* img_out;
    while(*stop != true) {
        __shared__ bool d_res;
        if(tid == 0){
            d_res = dequeue_gpu(&cpu_gpu_q[bid],&img_id, img_in, img_out);
        }
        __syncthreads();
        __threadfence_system();
        if(d_res == false){
            continue;
        }
        process_image_kernel(img_in, img_out);
        __syncthreads();
        __threadfence_system();
        if(tid == 0){
            enqueue_gpu(&gpu_cpu_q[bid],img_id);
        }  
    }
}

class queue_server : public image_processing_server {
private:
    int threadblock_number;
    bool *stop;
    producer_consumer_q *cpu_gpu_q;
    producer_consumer_q *gpu_cpu_q;
    __host__ bool enqueue_cpu(producer_consumer_q* q, int img_id, uchar* current_image, uchar* image_ptr) {
        int tail = q->queue_tail.load(cuda::memory_order_relaxed);
        if(((tail - q->queue_head.load(cuda::memory_order_acquire)) == QUEUE_SIZE)) {
            return false;
        }
        int index = q->queue_tail % QUEUE_SIZE;
        q->images[index] = current_image;
        q->image_ptrs[index] = image_ptr;
        q->image_index[index] = img_id;
        q->queue_tail.store((tail + 1), cuda::memory_order_release);
        return true;
    }
    
    __host__ bool dequeue_cpu(producer_consumer_q* q,int* img_id) {
        int head = q->queue_head.load(cuda::memory_order_relaxed);
        if(q->queue_tail.load(cuda::memory_order_acquire) == q->queue_head) {
            return false;
        }
        int index = q->queue_head % QUEUE_SIZE;
        *img_id = q->image_index[index];
        q->queue_head.store((head + 1), cuda::memory_order_release);
        return true;
    }

public:
    queue_server(int threads) {
        threadblock_number = query_device(threads);
        //allocation
        CUDA_CHECK( cudaMallocHost(&cpu_gpu_q, sizeof(producer_consumer_q) * threadblock_number) );
        CUDA_CHECK( cudaMallocHost(&gpu_cpu_q, sizeof(producer_consumer_q) * threadblock_number) );
        CUDA_CHECK( cudaMallocHost(&stop, sizeof(bool)) );
        //init queues 
        new (cpu_gpu_q) producer_consumer_q[threadblock_number];
        new (gpu_cpu_q) producer_consumer_q[threadblock_number];
        //stop flag
        *stop = false;
        //kernel invocation
        worker<<<threadblock_number, threads>>>(cpu_gpu_q, gpu_cpu_q, stop);
    }

    ~queue_server() override {
        *stop = true;
        CUDA_CHECK( cudaDeviceSynchronize() );
        if(gpu_cpu_q != NULL) CUDA_CHECK( cudaFreeHost(gpu_cpu_q) );
        if(cpu_gpu_q != NULL) CUDA_CHECK( cudaFreeHost(cpu_gpu_q) );
        if(cpu_gpu_q != NULL) CUDA_CHECK( cudaFreeHost(stop) );
        // if(stop != NULL) {
        //     stop->~atomic<bool>();
        //     CUDA_CHECK(cudaFreeHost(stop));
        // }
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override {
        for(int i = 0; i < threadblock_number; i++)
            if(enqueue_cpu(&cpu_gpu_q[i],img_id, img_in, img_out)) {
                return true;
            }
        return false;
    }

    bool dequeue(int *img_id) override {
        for(int i = 0; i < threadblock_number; i++)
            if(dequeue_cpu(&gpu_cpu_q[i],img_id)) {
                return true;
            }
        return false;
    }

};

std::unique_ptr<image_processing_server> create_queues_server(int threads) {
    return std::make_unique<queue_server>(threads);
}
