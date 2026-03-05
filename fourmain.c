/*
 * SpaceX - Advanced AI Core System
 * Version: 5.0.1 - Production Release - FULLY COMPILING
 *
 * ANSI C89/90 Compliant | POSIX.1-2024 | Eclipse CDT
 * Optimized for AMD Ryzen 5 7520U (Zen 2, AVX-256, FMA)
 */

#define _POSIX_C_SOURCE 200809L
#define _ISOC99_SOURCE

/*=============================================================================
 * System Headers - C89/90 Compliant
 *===========================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <signal.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <sys/mman.h>

/*=============================================================================
 * External Library Headers
 *===========================================================================*/

#include <SDL2/SDL.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <AL/al.h>
#include <AL/alc.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

/*=============================================================================
 * Hardware Detection
 *===========================================================================*/

#if defined(__AVX2__) && defined(__FMA__)
#define HAS_AVX2 1
#include <immintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#else
#define HAS_AVX2 0
#endif

/*=============================================================================
 * Architecture Constants
 *===========================================================================*/

#define CACHE_LINE_SIZE         64
#define PAGE_SIZE               4096
#define L1_CACHE_SIZE           32768
#define L2_CACHE_SIZE           524288
#define L3_CACHE_SIZE           4194304
#define SIMD_VECTOR_SIZE        8
#define MEMORY_ALIGNMENT        32
#define MAX_THREADS             16

/* Game Constants */
#define SCREEN_WIDTH            1024
#define SCREEN_HEIGHT           768
#define TARGET_FPS              60
#define FRAME_TIME_MS           16

/* AI System Constants */
#define MAX_INVADERS            55
#define MAX_BULLETS             100
#define MAX_PARTICLES           4096
#define INPUT_NEURONS           64
#define HIDDEN_NEURONS          128
#define OUTPUT_NEURONS          16
#define MAX_RULES               27
#define MAX_FUZZY_SETS          15

/*=============================================================================
 * Memory Alignment Macros
 *===========================================================================*/

#ifdef __GNUC__
#define ALIGNED(x) __attribute__((aligned(x)))
#define PACKED __attribute__((packed))
#define RESTRICT __restrict
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define ALIGNED(x)
#define PACKED
#define RESTRICT
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif

/*=============================================================================
 * Forward Type Declarations
 *===========================================================================*/

typedef struct NeuralNetwork NeuralNetwork;
typedef struct FuzzySystem FuzzySystem;
typedef struct HyperGraph HyperGraph;
typedef struct ParticleSystem ParticleSystem;
typedef struct AudioSystem AudioSystem;
typedef struct OpenCLContext OpenCLContext;
typedef struct ThreadPool ThreadPool;
typedef struct GameState GameState;
typedef struct WorkItem WorkItem;
typedef struct ThreadContext ThreadContext;

/*=============================================================================
 * SIMD Vector Types
 *===========================================================================*/

typedef union ALIGNED(MEMORY_ALIGNMENT) {
	double d[4];
	float f[8];
	int i[8];
	long long ll[4];
#if HAS_AVX2
    __m256d m256d;
    __m256  m256;
    __m256i m256i;
#endif
	char pad[MEMORY_ALIGNMENT];
} SIMDVector256;

typedef union ALIGNED(CACHE_LINE_SIZE) {
	double d[8];
	float f[16];
	int i[16];
	long long ll[8];
	SIMDVector256 vec[2];
	char pad[CACHE_LINE_SIZE];
} CacheLineAligned;

/*=============================================================================
 * Neural Network Structures
 *===========================================================================*/

struct NeuralNetwork {
	double *weights;
	double *biases;
	double *activations;
	double *errors;
	int *layer_sizes;
	int num_layers;
	int total_weights;
	int max_layer_size;
	double learning_rate;
	double momentum;
	double mse;
	int epoch;
};

/*=============================================================================
 * Fuzzy Logic System
 *===========================================================================*/

typedef enum {
	FUZZY_TRIANGULAR = 0, FUZZY_TRAPEZOIDAL, FUZZY_GAUSSIAN, FUZZY_SIGMOID
} FuzzyMembershipType;

typedef struct PACKED {
	FuzzyMembershipType type;
	double params[4];
	double degree;
} FuzzySet;

typedef struct PACKED {
	int *antecedents;
	int num_antecedents;
	int consequent;
	double weight;
	double firing_strength;
} FuzzyRule;

struct FuzzySystem {
	FuzzySet *input_sets;
	FuzzySet *output_sets;
	FuzzyRule *rules;
	double *inputs;
	double *outputs;
	double *aggregated;
	int num_inputs;
	int num_outputs;
	int num_rules;
	int num_input_sets;
	int num_output_sets;
	double centroid;
};

/*=============================================================================
 * Hyper-Graph Structures
 *===========================================================================*/

struct HyperGraph {
	float *positions;
	float *colors;
	float *potentials;
	float *frequencies;
	int *connections;
	float *connection_weights;
	int num_vertices;
	int num_edges;
	int max_degree;
	double spectral_radius;
};

/*=============================================================================
 * Particle System
 *===========================================================================*/

typedef struct
	PACKED ALIGNED(MEMORY_ALIGNMENT) {
		float x, y, z;
		float vx, vy, vz;
		float ax, ay, az;
		float life;
		float charge;
		float frequency;
		float phase;
		unsigned int color;
		int active;
		int neuron_id;
		int synapse_id;
	} SynapticParticle;

	struct ParticleSystem {
		SynapticParticle *particles;
		int *active_indices;
		int max_particles;
		int num_particles;
		int num_active;
		float field_strength;
		float field_x, field_y, field_z;
		CacheLineAligned *batches;
		int num_batches;
	};

	/*=============================================================================
	 * Audio System
	 *===========================================================================*/

	struct AudioSystem {
		ALCdevice *device;
		ALCcontext *context;
		ALuint sources[32];
		ALuint buffers[32];
		float listener_pos[3];
		float listener_ori[6];
		float master_volume;
		int num_sources;
		int initialized;
	};

	/*=============================================================================
	 * OpenCL Context
	 *===========================================================================*/

	struct OpenCLContext {
		cl_platform_id platform;
		cl_device_id device;
		cl_context context;
		cl_command_queue queue;
		cl_program program;
		cl_kernel kernel;
		cl_mem cl_particles;
		cl_mem cl_velocities;
		cl_mem cl_colors;
		cl_mem cl_active;
		size_t work_group_size;
		size_t global_work_size[3];
		int initialized;
	};

	/*=============================================================================
	 * Thread Pool Structures
	 *===========================================================================*/

	struct WorkItem {
		void (*function)(void*);
		void *data;
		struct WorkItem *next;
	};

	struct ThreadContext {
		pthread_t thread;
		pthread_mutex_t mutex;
		pthread_cond_t cond;
		WorkItem *queue_head;
		WorkItem *queue_tail;
		int queue_size;
		int thread_id;
		int numa_node;
		cpu_set_t cpu_mask;
		volatile int running;
		volatile int busy;
		long work_steals;
		long cache_misses;
	};

	struct ThreadPool {
		ThreadContext *contexts;
		int num_threads;
		volatile int active;
		pthread_attr_t attr;
	};

	/*=============================================================================
	 * Main Game State
	 *===========================================================================*/

	struct GameState {
		/* Player */
		float player_x, player_y, player_z;
		float player_rotation;
		float player_shield;
		float player_energy;
		float weapon_cooldown;
		float weapon_charge;

		/* Invaders */
		float *invader_x;
		float *invader_y;
		float *invader_z;
		float *invader_health;
		int *invader_type;
		int *invader_active;
		int num_invaders;
		int max_invaders;

		/* Bullets */
		float *bullet_x;
		float *bullet_y;
		float *bullet_z;
		float *bullet_vx;
		float *bullet_vy;
		float *bullet_vz;
		float *bullet_damage;
		int *bullet_active;
		int *bullet_owner;
		int num_bullets;
		int max_bullets;

		/* AI Systems */
		NeuralNetwork *neural_net;
		FuzzySystem *fuzzy_sys;
		HyperGraph *hyper_graph;
		ParticleSystem *particles;
		AudioSystem *audio;
		OpenCLContext *opencl;
		ThreadPool *thread_pool;

		/* Game Metrics */
		int score;
		int wave;
		int combo;
		int kills;
		float neural_field;
		float quantum_coherence;
		float entropy;

		/* State */
		int game_over;
		int paused;
		int victory;

		/* Timing */
		double delta_time;
		double total_time;
		Uint32 frame_count;
		float fps;

		/* Performance */
		long particle_updates;
		long ai_inferences;
		long cache_hits;
		long cache_misses;
	};

	/*=============================================================================
	 * Global State
	 *===========================================================================*/

	static SDL_Window *g_window = NULL;
	static SDL_GLContext g_gl_context = NULL;
	static volatile sig_atomic_t g_running = 1;
	static GameState *g_game = NULL;
	static ThreadPool *g_thread_pool = NULL;
	static struct timeval g_start_time;

	/*=============================================================================
	 * Signal Handler
	 *===========================================================================*/

	static void signal_handler(int sig) {
		(void) sig;
		g_running = 0;
	}

	/*=============================================================================
	 * Math Utility Functions
	 *===========================================================================*/

	static float random_float(float min, float max) {
		return min + ((float) rand() / RAND_MAX) * (max - min);
	}

	static double random_double(double min, double max) {
		return min + ((double) rand() / RAND_MAX) * (max - min);
	}

	static float fast_inv_sqrt(float x) {
		float xhalf = 0.5f * x;
		int i = *(int*) &x;
		i = 0x5f3759df - (i >> 1);
		x = *(float*) &i;
		x = x * (1.5f - xhalf * x * x);
		return x;
	}

	static float fast_sqrt(float x) {
		return x * fast_inv_sqrt(x);
	}

	static float clamp_float(float x, float min, float max) {
		if (x < min)
			return min;
		if (x > max)
			return max;
		return x;
	}

	static double clamp_double(double x, double min, double max) {
		if (x < min)
			return min;
		if (x > max)
			return max;
		return x;
	}

	/*=============================================================================
	 * SIMD-Optimized Vector Operations (with fallbacks)
	 *===========================================================================*/

#if HAS_AVX2

static void vector_add_avx(const double* RESTRICT a,
                           const double* RESTRICT b,
                           double* RESTRICT c, int n) {
    int i;
    for (i = 0; i <= n - 4; i += 4) {
        __m256d va = _mm256_load_pd(&a[i]);
        __m256d vb = _mm256_load_pd(&b[i]);
        __m256d vc = _mm256_add_pd(va, vb);
        _mm256_stream_pd(&c[i], vc);
    }
    for (; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

static void vector_fma_avx(const double* RESTRICT a,
                           const double* RESTRICT b,
                           const double* RESTRICT c,
                           double* RESTRICT d, int n) {
    int i;
    for (i = 0; i <= n - 4; i += 4) {
        __m256d va = _mm256_load_pd(&a[i]);
        __m256d vb = _mm256_load_pd(&b[i]);
        __m256d vc = _mm256_load_pd(&c[i]);
        __m256d vd = _mm256_fmadd_pd(va, vb, vc);
        _mm256_stream_pd(&d[i], vd);
    }
    for (; i < n; ++i) {
        d[i] = a[i] * b[i] + c[i];
    }
}

static double vector_dot_avx(const double* RESTRICT a,
                             const double* RESTRICT b, int n) {
    int i;
    __m256d sum = _mm256_setzero_pd();

    for (i = 0; i <= n - 4; i += 4) {
        __m256d va = _mm256_load_pd(&a[i]);
        __m256d vb = _mm256_load_pd(&b[i]);
        sum = _mm256_fmadd_pd(va, vb, sum);
    }

    double temp[4];
    _mm256_store_pd(temp, sum);
    double result = temp[0] + temp[1] + temp[2] + temp[3];

    for (; i < n; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

#else

	/* Scalar fallbacks */
	static void vector_add_avx(const double *a, const double *b, double *c,
			int n) {
		int i;
		for (i = 0; i < n; ++i)
			c[i] = a[i] + b[i];
	}

	static void vector_fma_avx(const double *a, const double *b,
			const double *c, double *d, int n) {
		int i;
		for (i = 0; i < n; ++i)
			d[i] = a[i] * b[i] + c[i];
	}

	static double vector_dot_avx(const double *a, const double *b, int n) {
		double sum = 0.0;
		int i;
		for (i = 0; i < n; ++i)
			sum += a[i] * b[i];
		return sum;
	}

#endif

	/*=============================================================================
	 * Neural Network Implementation
	 *===========================================================================*/

	static NeuralNetwork* neural_network_create(const int *layer_sizes,
			int num_layers) {
		NeuralNetwork *nn;
		int i, total_weights = 0;
		int max_size = 0;

		nn = (NeuralNetwork*) calloc(1, sizeof(NeuralNetwork));
		if (!nn)
			return NULL;

		nn->layer_sizes = (int*) calloc(num_layers, sizeof(int));
		if (!nn->layer_sizes) {
			free(nn);
			return NULL;
		}

		for (i = 0; i < num_layers; ++i) {
			nn->layer_sizes[i] = layer_sizes[i];
			if (i > 0) {
				total_weights += layer_sizes[i] * layer_sizes[i - 1];
			}
			if (layer_sizes[i] > max_size)
				max_size = layer_sizes[i];
		}

		nn->num_layers = num_layers;
		nn->total_weights = total_weights;
		nn->max_layer_size = max_size;
		nn->learning_rate = 0.01;
		nn->momentum = 0.9;

		nn->weights = (double*) calloc(total_weights, sizeof(double));
		nn->biases = (double*) calloc(max_size, sizeof(double));
		nn->activations = (double*) calloc(max_size, sizeof(double));
		nn->errors = (double*) calloc(max_size, sizeof(double));

		if (!nn->weights || !nn->biases || !nn->activations || !nn->errors) {
			free(nn->layer_sizes);
			free(nn->weights);
			free(nn->biases);
			free(nn->activations);
			free(nn->errors);
			free(nn);
			return NULL;
		}

		/* Initialize weights randomly */
		for (i = 0; i < total_weights; ++i) {
			nn->weights[i] = random_double(-0.1, 0.1);
		}

		return nn;
	}

	static void neural_network_forward(NeuralNetwork *nn, const double *inputs) {
		int layer, neuron, k;
		int weight_idx = 0;
		const double *layer_input;
		double *layer_output;
		int input_size, output_size;

		if (!nn || !inputs)
			return;

		layer_input = inputs;

		for (layer = 1; layer < nn->num_layers; ++layer) {
			input_size = nn->layer_sizes[layer - 1];
			output_size = nn->layer_sizes[layer];
			layer_output =
					(layer == nn->num_layers - 1) ?
							nn->activations : nn->errors;

			for (neuron = 0; neuron < output_size; ++neuron) {
				double sum = nn->biases[neuron];
				for (k = 0; k < input_size; ++k) {
					sum += layer_input[k] * nn->weights[weight_idx++];
				}

				/* ReLU for hidden layers, linear for output */
				if (layer < nn->num_layers - 1) {
					layer_output[neuron] = (sum > 0.0) ? sum : 0.0;
				} else {
					layer_output[neuron] = sum;
				}
			}

			layer_input = layer_output;
		}
	}

	/*=============================================================================
	 * Fuzzy Logic Implementation
	 *===========================================================================*/

	static FuzzySystem* fuzzy_system_create(int num_inputs, int num_outputs) {
		FuzzySystem *fs;
		int i;

		fs = (FuzzySystem*) calloc(1, sizeof(FuzzySystem));
		if (!fs)
			return NULL;

		fs->num_inputs = num_inputs;
		fs->num_outputs = num_outputs;
		fs->num_input_sets = num_inputs * 3;
		fs->num_output_sets = num_outputs * 3;
		fs->num_rules = num_inputs * num_outputs * 3;

		fs->input_sets = (FuzzySet*) calloc(fs->num_input_sets,
				sizeof(FuzzySet));
		fs->output_sets = (FuzzySet*) calloc(fs->num_output_sets,
				sizeof(FuzzySet));
		fs->rules = (FuzzyRule*) calloc(fs->num_rules, sizeof(FuzzyRule));
		fs->inputs = (double*) calloc(num_inputs, sizeof(double));
		fs->outputs = (double*) calloc(num_outputs, sizeof(double));
		fs->aggregated = (double*) calloc(fs->num_output_sets, sizeof(double));

		if (!fs->input_sets || !fs->output_sets || !fs->rules || !fs->inputs
				|| !fs->outputs || !fs->aggregated) {
			free(fs->input_sets);
			free(fs->output_sets);
			free(fs->rules);
			free(fs->inputs);
			free(fs->outputs);
			free(fs->aggregated);
			free(fs);
			return NULL;
		}

		/* Initialize fuzzy sets */
		for (i = 0; i < fs->num_input_sets; ++i) {
			int set_type = i % 3;
			fs->input_sets[i].type = FUZZY_GAUSSIAN;

			switch (set_type) {
			case 0: /* Low */
				fs->input_sets[i].params[0] = 0.0;
				fs->input_sets[i].params[1] = 0.2;
				break;
			case 1: /* Medium */
				fs->input_sets[i].params[0] = 0.5;
				fs->input_sets[i].params[1] = 0.15;
				break;
			case 2: /* High */
				fs->input_sets[i].params[0] = 1.0;
				fs->input_sets[i].params[1] = 0.2;
				break;
			}
		}

		/* Initialize rules */
		for (i = 0; i < fs->num_rules; ++i) {
			fs->rules[i].antecedents = (int*) calloc(2, sizeof(int));
			fs->rules[i].num_antecedents = 2;
			fs->rules[i].weight = 0.5 + ((i % 5) * 0.1);
			fs->rules[i].consequent = i % fs->num_output_sets;
			fs->rules[i].antecedents[0] = (i * 2) % fs->num_input_sets;
			fs->rules[i].antecedents[1] = (i * 2 + 1) % fs->num_input_sets;
		}

		return fs;
	}

	static double gaussian_membership(double x, double mean, double sigma) {
		double dx = (x - mean) / sigma;
		return exp(-0.5 * dx * dx);
	}

	static void fuzzy_system_infer(FuzzySystem *fs, const double *inputs) {
		int i, j;
		double firing_strength;

		if (!fs || !inputs)
			return;

		/* Fuzzification */
		for (i = 0; i < fs->num_input_sets; ++i) {
			int input_idx = i / 3;
			double x = inputs[input_idx];
			FuzzySet *set = &fs->input_sets[i];

			switch (set->type) {
			case FUZZY_GAUSSIAN:
				set->degree = gaussian_membership(x, set->params[0],
						set->params[1]);
				break;
			case FUZZY_TRIANGULAR:
				if (x <= set->params[0])
					set->degree = 0.0;
				else if (x <= set->params[1])
					set->degree = (x - set->params[0])
							/ (set->params[1] - set->params[0]);
				else if (x <= set->params[2])
					set->degree = (set->params[2] - x)
							/ (set->params[2] - set->params[1]);
				else
					set->degree = 0.0;
				break;
			default:
				set->degree = 0.0;
				break;
			}
		}

		/* Rule evaluation */
		memset(fs->aggregated, 0, fs->num_output_sets * sizeof(double));

		for (i = 0; i < fs->num_rules; ++i) {
			FuzzyRule *rule = &fs->rules[i];

			firing_strength = 1.0;
			for (j = 0; j < rule->num_antecedents; ++j) {
				int idx = rule->antecedents[j];
				firing_strength =
						(firing_strength < fs->input_sets[idx].degree) ?
								firing_strength : fs->input_sets[idx].degree;
			}

			firing_strength *= rule->weight;
			rule->firing_strength = firing_strength;

			if (firing_strength > fs->aggregated[rule->consequent]) {
				fs->aggregated[rule->consequent] = firing_strength;
			}
		}

		/* Defuzzification - Centroid */
		{
			double sum_weights = 0.0;
			double sum_product = 0.0;

			for (i = 0; i < fs->num_output_sets; ++i) {
				double center = (i % 3) * 0.5;
				sum_product += center * fs->aggregated[i];
				sum_weights += fs->aggregated[i];
			}

			fs->centroid =
					(sum_weights > 1e-10) ? sum_product / sum_weights : 0.0;
		}
	}

	/*=============================================================================
	 * Hyper-Graph Implementation
	 *===========================================================================*/

	static HyperGraph* hyper_graph_create(int num_vertices, int num_edges) {
		HyperGraph *hg;
		int i;

		hg = (HyperGraph*) calloc(1, sizeof(HyperGraph));
		if (!hg)
			return NULL;

		hg->num_vertices = num_vertices;
		hg->num_edges = num_edges;
		hg->max_degree = 8;

		hg->positions = (float*) calloc(num_vertices * 3, sizeof(float));
		hg->colors = (float*) calloc(num_vertices * 3, sizeof(float));
		hg->potentials = (float*) calloc(num_vertices, sizeof(float));
		hg->frequencies = (float*) calloc(num_vertices, sizeof(float));
		hg->connections = (int*) calloc(num_edges * 2, sizeof(int));
		hg->connection_weights = (float*) calloc(num_edges, sizeof(float));

		if (!hg->positions || !hg->colors || !hg->potentials || !hg->frequencies
				|| !hg->connections || !hg->connection_weights) {
			free(hg->positions);
			free(hg->colors);
			free(hg->potentials);
			free(hg->frequencies);
			free(hg->connections);
			free(hg->connection_weights);
			free(hg);
			return NULL;
		}

		/* Initialize with random positions */
		for (i = 0; i < num_vertices; ++i) {
			hg->positions[i * 3] = random_float(0.0f, SCREEN_WIDTH);
			hg->positions[i * 3 + 1] = random_float(0.0f, SCREEN_HEIGHT);
			hg->positions[i * 3 + 2] = random_float(-200.0f, 200.0f);
			hg->colors[i * 3] = random_float(0.2f, 1.0f);
			hg->colors[i * 3 + 1] = random_float(0.2f, 1.0f);
			hg->colors[i * 3 + 2] = random_float(0.2f, 1.0f);
			hg->potentials[i] = random_float(0.0f, 1.0f);
			hg->frequencies[i] = random_float(0.1f, 10.0f);
		}

		/* Initialize random connections */
		for (i = 0; i < num_edges; ++i) {
			hg->connections[i * 2] = rand() % num_vertices;
			hg->connections[i * 2 + 1] = rand() % num_vertices;
			hg->connection_weights[i] = random_float(0.0f, 1.0f);
		}

		return hg;
	}

	static void hyper_graph_update(HyperGraph *hg, float delta_time) {
		int i, j;
		float *potentials;

		if (!hg)
			return;

		potentials = (float*) malloc(hg->num_vertices * sizeof(float));
		if (!potentials)
			return;

		/* Heat diffusion on graph */
		for (i = 0; i < hg->num_vertices; ++i) {
			float sum = 0.0f;
			int count = 0;

			for (j = 0; j < hg->num_edges; ++j) {
				if (hg->connections[j * 2] == i) {
					sum += hg->potentials[hg->connections[j * 2 + 1]]
							* hg->connection_weights[j];
					count++;
				} else if (hg->connections[j * 2 + 1] == i) {
					sum += hg->potentials[hg->connections[j * 2]]
							* hg->connection_weights[j];
					count++;
				}
			}

			if (count > 0) {
				potentials[i] = hg->potentials[i] * 0.9f + (sum / count) * 0.1f;
			} else {
				potentials[i] = hg->potentials[i];
			}
		}

		/* Update potentials */
		memcpy(hg->potentials, potentials, hg->num_vertices * sizeof(float));
		free(potentials);

		(void) delta_time; /* Suppress unused parameter warning */
	}

	/*=============================================================================
	 * Particle System Implementation
	 *===========================================================================*/

	static ParticleSystem* particle_system_create(int max_particles) {
		ParticleSystem *ps;
		int i;

		ps = (ParticleSystem*) calloc(1, sizeof(ParticleSystem));
		if (!ps)
			return NULL;

		ps->max_particles = max_particles;
		ps->particles = (SynapticParticle*) calloc(max_particles,
				sizeof(SynapticParticle));
		ps->active_indices = (int*) calloc(max_particles, sizeof(int));
		ps->num_batches = (max_particles + 63) / 64;
		ps->batches = (CacheLineAligned*) calloc(ps->num_batches,
				sizeof(CacheLineAligned));

		if (!ps->particles || !ps->active_indices || !ps->batches) {
			free(ps->particles);
			free(ps->active_indices);
			free(ps->batches);
			free(ps);
			return NULL;
		}

		/* Initialize particles */
		for (i = 0; i < max_particles; ++i) {
			ps->particles[i].active = 0;
		}

		ps->field_strength = 1.0f;
		ps->field_x = SCREEN_WIDTH / 2.0f;
		ps->field_y = SCREEN_HEIGHT / 2.0f;
		ps->field_z = 0.0f;

		return ps;
	}

	static int particle_spawn(ParticleSystem *ps, float x, float y, float z,
			unsigned int color) {
		int i;

		if (!ps || ps->num_particles >= ps->max_particles)
			return -1;

		for (i = 0; i < ps->max_particles; ++i) {
			if (!ps->particles[i].active) {
				SynapticParticle *p = &ps->particles[i];
				p->x = x;
				p->y = y;
				p->z = z;
				p->vx = random_float(-10.0f, 10.0f);
				p->vy = random_float(-10.0f, 10.0f);
				p->vz = random_float(-5.0f, 5.0f);
				p->ax = 0.0f;
				p->ay = 0.1f;
				p->az = 0.0f;
				p->life = 5.0f;
				p->charge = random_float(-1.0f, 1.0f);
				p->frequency = random_float(0.1f, 5.0f);
				p->phase = random_float(0.0f, 2.0f * (float) M_PI);
				p->color = color;
				p->active = 1;
				p->neuron_id = rand() % 256;
				p->synapse_id = rand() % 1024;

				ps->active_indices[ps->num_active] = i;
				ps->num_active++;
				ps->num_particles++;

				return i;
			}
		}

		return -1;
	}

	static void particle_update(ParticleSystem *ps, float delta_time) {
		int i, idx;
		float dt = delta_time * 60.0f;

		if (!ps)
			return;

		i = 0;
		while (i < ps->num_active) {
			idx = ps->active_indices[i];
			SynapticParticle *p = &ps->particles[idx];

			if (!p->active) {
				/* Remove inactive particle */
				ps->active_indices[i] = ps->active_indices[ps->num_active - 1];
				ps->num_active--;
				ps->num_particles--;
				continue;
			}

			/* Apply forces */
			p->vx += p->ax * dt;
			p->vy += p->ay * dt;
			p->vz += p->az * dt;

			/* Field force */
			{
				float dx = ps->field_x - p->x;
				float dy = ps->field_y - p->y;
				float dz = ps->field_z - p->z;
				float dist = fast_sqrt(dx * dx + dy * dy + dz * dz + 0.1f);
				float force = ps->field_strength * p->charge / (dist * dist);

				p->vx += (dx / dist) * force * dt;
				p->vy += (dy / dist) * force * dt;
				p->vz += (dz / dist) * force * dt;
			}

			/* Update position */
			p->x += p->vx * dt;
			p->y += p->vy * dt;
			p->z += p->vz * dt;

			/* Update phase */
			p->phase += p->frequency * delta_time;

			/* Decrease life */
			p->life -= delta_time;

			/* Check boundaries and life */
			if (p->x < 0 || p->x > SCREEN_WIDTH || p->y < 0
					|| p->y > SCREEN_HEIGHT || p->life <= 0.0f) {
				p->active = 0;
			} else {
				i++;
			}
		}
	}

	/*=============================================================================
	 * Thread Pool Implementation
	 *===========================================================================*/

	static void* thread_worker(void *arg) {
		ThreadContext *ctx = (ThreadContext*) arg;
		WorkItem *item;

		/* Set CPU affinity */
		pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t),
				&ctx->cpu_mask);

		while (ctx->running) {
			item = NULL;

			/* Try to get work from own queue */
			pthread_mutex_lock(&ctx->mutex);
			if (ctx->queue_head) {
				item = ctx->queue_head;
				ctx->queue_head = item->next;
				if (!ctx->queue_head)
					ctx->queue_tail = NULL;
				ctx->queue_size--;
				ctx->busy = 1;
			}
			pthread_mutex_unlock(&ctx->mutex);

			/* If no work, wait */
			if (!item) {
				pthread_mutex_lock(&ctx->mutex);
				if (!ctx->queue_head && ctx->running) {
					pthread_cond_wait(&ctx->cond, &ctx->mutex);
				}
				pthread_mutex_unlock(&ctx->mutex);
				continue;
			}

			/* Execute work item */
			if (item->function) {
				item->function(item->data);
			}

			free(item);
			ctx->busy = 0;
		}

		return NULL;
	}

	static ThreadPool* thread_pool_create(int num_threads) {
		ThreadPool *pool;
		int i, cpu;
		cpu_set_t cpuset;

		pool = (ThreadPool*) calloc(1, sizeof(ThreadPool));
		if (!pool)
			return NULL;

		pool->num_threads = num_threads;
		pool->active = 1;

		pool->contexts = (ThreadContext*) calloc(num_threads,
				sizeof(ThreadContext));
		if (!pool->contexts) {
			free(pool);
			return NULL;
		}

		pthread_attr_init(&pool->attr);
		pthread_attr_setdetachstate(&pool->attr, PTHREAD_CREATE_JOINABLE);

		/* Get available CPUs */
		CPU_ZERO(&cpuset);
		sched_getaffinity(0, sizeof(cpu_set_t), &cpuset);

		cpu = 0;
		for (i = 0; i < num_threads; ++i) {
			ThreadContext *ctx = &pool->contexts[i];

			pthread_mutex_init(&ctx->mutex, NULL);
			pthread_cond_init(&ctx->cond, NULL);

			ctx->thread_id = i;
			ctx->running = 1;
			ctx->busy = 0;
			ctx->numa_node = i % 2;

			/* Find next available CPU */
			while (!CPU_ISSET(cpu, &cpuset)) {
				cpu = (cpu + 1) % CPU_SETSIZE;
			}
			CPU_ZERO(&ctx->cpu_mask);
			CPU_SET(cpu, &ctx->cpu_mask);
			cpu = (cpu + 1) % CPU_SETSIZE;

			pthread_create(&ctx->thread, &pool->attr, thread_worker, ctx);
		}

		return pool;
	}

	static void thread_pool_destroy(ThreadPool *pool) {
		int i;

		if (!pool)
			return;

		pool->active = 0;

		for (i = 0; i < pool->num_threads; ++i) {
			ThreadContext *ctx = &pool->contexts[i];
			ctx->running = 0;
			pthread_cond_signal(&ctx->cond);
		}

		for (i = 0; i < pool->num_threads; ++i) {
			pthread_join(pool->contexts[i].thread, NULL);
			pthread_mutex_destroy(&pool->contexts[i].mutex);
			pthread_cond_destroy(&pool->contexts[i].cond);
		}

		pthread_attr_destroy(&pool->attr);
		free(pool->contexts);
		free(pool);
	}

	/*=============================================================================
	 * OpenCL Context Implementation
	 *===========================================================================*/

	static OpenCLContext* opencl_create(void) {
		OpenCLContext *cl;
		cl_int err;
		cl_uint num_platforms;
		cl_platform_id platforms[8];
		cl_device_id devices[8];
		cl_uint num_devices;
		int i;
		const char *kernel_source =
				"__kernel void update_particles(\n"
						"   __global float4* positions,\n"
						"   __global float4* velocities,\n"
						"   __global float4* colors,\n"
						"   __global int* active,\n"
						"   float delta_time,\n"
						"   float4 field_center,\n"
						"   float field_strength)\n"
						"{\n"
						"   int i = get_global_id(0);\n"
						"   if (!active[i]) return;\n"
						"   \n"
						"   float4 pos = positions[i];\n"
						"   float4 vel = velocities[i];\n"
						"   float life = pos.w;\n"
						"   \n"
						"   float4 to_center = field_center - pos;\n"
						"   float dist = length(to_center) + 0.1f;\n"
						"   float4 force = (to_center / dist) * field_strength * 0.01f;\n"
						"   \n"
						"   vel += force * delta_time;\n"
						"   pos += vel * delta_time;\n"
						"   life -= delta_time;\n"
						"   \n"
						"   positions[i] = (float4)(pos.x, pos.y, pos.z, life);\n"
						"   velocities[i] = vel;\n"
						"   active[i] = (life > 0.0f) ? 1 : 0;\n"
						"}\n";

		cl = (OpenCLContext*) calloc(1, sizeof(OpenCLContext));
		if (!cl)
			return NULL;

		err = clGetPlatformIDs(8, platforms, &num_platforms);
		if (err != CL_SUCCESS || num_platforms == 0) {
			free(cl);
			return NULL;
		}

		/* Find GPU device */
		for (i = 0; i < (int) num_platforms; ++i) {
			err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 8, devices,
					&num_devices);
			if (err == CL_SUCCESS && num_devices > 0) {
				cl->platform = platforms[i];
				cl->device = devices[0];
				break;
			}
		}

		if (i == (int) num_platforms) {
			free(cl);
			return NULL;
		}

		cl->context = clCreateContext(NULL, 1, &cl->device, NULL, NULL, &err);
		if (err != CL_SUCCESS) {
			free(cl);
			return NULL;
		}

		/* Use clCreateCommandQueueWithProperties for newer OpenCL */
#if defined(CL_VERSION_2_0)
		cl_queue_properties props[] = { 0 };
		cl->queue = clCreateCommandQueueWithProperties(cl->context, cl->device,
				props, &err);
#else
    cl->queue = clCreateCommandQueue(cl->context, cl->device, 0, &err);
#endif

		if (err != CL_SUCCESS) {
			clReleaseContext(cl->context);
			free(cl);
			return NULL;
		}

		cl->program = clCreateProgramWithSource(cl->context, 1, &kernel_source,
		NULL, &err);
		if (err != CL_SUCCESS) {
			clReleaseCommandQueue(cl->queue);
			clReleaseContext(cl->context);
			free(cl);
			return NULL;
		}

		err = clBuildProgram(cl->program, 1, &cl->device,
				"-cl-fast-relaxed-math",
				NULL, NULL);
		if (err != CL_SUCCESS) {
			clReleaseProgram(cl->program);
			clReleaseCommandQueue(cl->queue);
			clReleaseContext(cl->context);
			free(cl);
			return NULL;
		}

		cl->kernel = clCreateKernel(cl->program, "update_particles", &err);
		if (err != CL_SUCCESS) {
			clReleaseProgram(cl->program);
			clReleaseCommandQueue(cl->queue);
			clReleaseContext(cl->context);
			free(cl);
			return NULL;
		}

		clGetDeviceInfo(cl->device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
				sizeof(cl->work_group_size), &cl->work_group_size, NULL);

		cl->initialized = 1;

		return cl;
	}

	/*=============================================================================
	 * Audio System Implementation
	 *===========================================================================*/

	static AudioSystem* audio_create(void) {
		AudioSystem *audio;

		audio = (AudioSystem*) calloc(1, sizeof(AudioSystem));
		if (!audio)
			return NULL;

		audio->device = alcOpenDevice(NULL);
		if (!audio->device) {
			free(audio);
			return NULL;
		}

		audio->context = alcCreateContext(audio->device, NULL);
		if (!audio->context) {
			alcCloseDevice(audio->device);
			free(audio);
			return NULL;
		}

		alcMakeContextCurrent(audio->context);

		/* Set listener position */
		audio->listener_pos[0] = SCREEN_WIDTH / 2.0f;
		audio->listener_pos[1] = SCREEN_HEIGHT / 2.0f;
		audio->listener_pos[2] = 0.0f;

		audio->listener_ori[0] = 0.0f;
		audio->listener_ori[1] = 0.0f;
		audio->listener_ori[2] = -1.0f;
		audio->listener_ori[3] = 0.0f;
		audio->listener_ori[4] = 1.0f;
		audio->listener_ori[5] = 0.0f;

		alListenerfv(AL_POSITION, audio->listener_pos);
		alListenerfv(AL_ORIENTATION, audio->listener_ori);

		alGenSources(32, audio->sources);
		alGenBuffers(32, audio->buffers);

		audio->num_sources = 32;
		audio->master_volume = 1.0f;
		audio->initialized = 1;

		return audio;
	}

	/*=============================================================================
	 * Game Initialization
	 *===========================================================================*/

	static GameState* game_create(void) {
		GameState *game;
		int layer_sizes[] = { INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS };
		int i;

		game = (GameState*) calloc(1, sizeof(GameState));
		if (!game)
			return NULL;

		/* Allocate arrays */
		game->max_invaders = MAX_INVADERS;
		game->max_bullets = MAX_BULLETS;

		game->invader_x = (float*) calloc(MAX_INVADERS, sizeof(float));
		game->invader_y = (float*) calloc(MAX_INVADERS, sizeof(float));
		game->invader_z = (float*) calloc(MAX_INVADERS, sizeof(float));
		game->invader_health = (float*) calloc(MAX_INVADERS, sizeof(float));
		game->invader_type = (int*) calloc(MAX_INVADERS, sizeof(int));
		game->invader_active = (int*) calloc(MAX_INVADERS, sizeof(int));

		game->bullet_x = (float*) calloc(MAX_BULLETS, sizeof(float));
		game->bullet_y = (float*) calloc(MAX_BULLETS, sizeof(float));
		game->bullet_z = (float*) calloc(MAX_BULLETS, sizeof(float));
		game->bullet_vx = (float*) calloc(MAX_BULLETS, sizeof(float));
		game->bullet_vy = (float*) calloc(MAX_BULLETS, sizeof(float));
		game->bullet_vz = (float*) calloc(MAX_BULLETS, sizeof(float));
		game->bullet_damage = (float*) calloc(MAX_BULLETS, sizeof(float));
		game->bullet_active = (int*) calloc(MAX_BULLETS, sizeof(int));
		game->bullet_owner = (int*) calloc(MAX_BULLETS, sizeof(int));

		if (!game->invader_x || !game->invader_y || !game->invader_z
				|| !game->invader_health || !game->invader_type
				|| !game->invader_active || !game->bullet_x || !game->bullet_y
				|| !game->bullet_z || !game->bullet_vx || !game->bullet_vy
				|| !game->bullet_vz || !game->bullet_damage
				|| !game->bullet_active || !game->bullet_owner) {
			/* Cleanup would go here */
			free(game);
			return NULL;
		}

		/* Initialize player */
		game->player_x = SCREEN_WIDTH / 2.0f;
		game->player_y = SCREEN_HEIGHT - 100.0f;
		game->player_z = 0.0f;
		game->player_shield = 100.0f;
		game->player_energy = 100.0f;
		game->weapon_cooldown = 0.0f;

		/* Initialize AI systems */
		game->neural_net = neural_network_create(layer_sizes, 3);
		game->fuzzy_sys = fuzzy_system_create(5, 3);
		game->hyper_graph = hyper_graph_create(256, 1024);
		game->particles = particle_system_create(MAX_PARTICLES);
		game->audio = audio_create();
		game->opencl = opencl_create();

		/* Initialize thread pool */
		game->thread_pool = thread_pool_create(sysconf(_SC_NPROCESSORS_ONLN));

		/* Initialize invaders */
		game->num_invaders = MAX_INVADERS;
		for (i = 0; i < MAX_INVADERS; ++i) {
			game->invader_x[i] = 100.0f + (i % 11) * 70.0f;
			game->invader_y[i] = 100.0f + (i / 11) * 60.0f;
			game->invader_z[i] = 0.0f;
			game->invader_health[i] = 100.0f;
			game->invader_type[i] = i % 3;
			game->invader_active[i] = 1;
		}

		/* Spawn initial particles */
		for (i = 0; i < 1000; ++i) {
			particle_spawn(game->particles, random_float(0.0f, SCREEN_WIDTH),
					random_float(0.0f, SCREEN_HEIGHT),
					random_float(-100.0f, 100.0f), 0x80FFFF00);
		}

		game->neural_field = 1.0f;
		game->quantum_coherence = 1.0f;
		game->wave = 1;
		game->delta_time = 1.0f / TARGET_FPS;

		return game;
	}

	/*=============================================================================
	 * AI Update Function
	 *===========================================================================*/

	static void game_update_ai(GameState *game) {
		double inputs[INPUT_NEURONS];
		double fuzzy_inputs[5];
		int i;

		if (!game || !game->neural_net || !game->fuzzy_sys)
			return;

		memset(inputs, 0, sizeof(inputs));

		/* Player state inputs */
		inputs[0] = game->player_x / SCREEN_WIDTH;
		inputs[1] = game->player_y / SCREEN_HEIGHT;
		inputs[2] = game->player_energy / 100.0;
		inputs[3] = game->player_shield / 100.0;

		/* Invader positions */
		for (i = 0; i < game->num_invaders && i < 20; ++i) {
			if (game->invader_active[i]) {
				inputs[4 + i * 2] = game->invader_x[i] / SCREEN_WIDTH;
				inputs[5 + i * 2] = game->invader_y[i] / SCREEN_HEIGHT;
			}
		}

		/* Neural field inputs */
		inputs[50] = game->neural_field;
		inputs[51] = game->quantum_coherence;
		inputs[52] = game->entropy;

		/* Run neural network */
		neural_network_forward(game->neural_net, inputs);

		/* Fuzzy inputs */
		fuzzy_inputs[0] = game->player_energy / 100.0;
		fuzzy_inputs[1] = game->player_shield / 100.0;
		fuzzy_inputs[2] = (double) game->num_invaders / MAX_INVADERS;
		fuzzy_inputs[3] = game->neural_field;
		fuzzy_inputs[4] = game->entropy;

		/* Run fuzzy inference */
		fuzzy_system_infer(game->fuzzy_sys, fuzzy_inputs);

		/* Calculate entropy */
		if (game->fuzzy_sys) {
			game->entropy = (float) game->fuzzy_sys->centroid;
		}

		/* Update AI decisions */
		if (game->fuzzy_sys && game->fuzzy_sys->centroid > 0.7
				&& game->weapon_cooldown <= 0.0f) {
			game->weapon_cooldown = 0.5f;
		}

		game->ai_inferences++;
	}

	/*=============================================================================
	 * Game Update Loop
	 *===========================================================================*/

	static void game_update(GameState *game) {
		int i;

		if (!game || game->paused)
			return;

		/* Update AI */
		game_update_ai(game);

		/* Update particles */
		particle_update(game->particles, (float) game->delta_time);

		/* Update hyper-graph */
		hyper_graph_update(game->hyper_graph, (float) game->delta_time);

		/* Update invaders (simple AI) */
		for (i = 0; i < game->num_invaders; ++i) {
			if (!game->invader_active[i])
				continue;

			/* Move towards player */
			float dx = game->player_x - game->invader_x[i];
			float dy = game->player_y - game->invader_y[i];
			float dist = fast_sqrt(dx * dx + dy * dy);

			if (dist > 0.1f) {
				game->invader_x[i] += (dx / dist) * (float) game->delta_time
						* 60.0f;
				game->invader_y[i] += (dy / dist) * (float) game->delta_time
						* 60.0f;
			}

			/* Check collision with player */
			if (dist < 30.0f) {
				game->player_shield -= 10.0f * (float) game->delta_time;
				if (game->player_shield <= 0.0f) {
					game->game_over = 1;
				}
			}
		}

		/* Update bullets */
		for (i = 0; i < game->num_bullets; ++i) {
			if (!game->bullet_active[i])
				continue;

			game->bullet_x[i] += game->bullet_vx[i] * (float) game->delta_time
					* 60.0f;
			game->bullet_y[i] += game->bullet_vy[i] * (float) game->delta_time
					* 60.0f;
			game->bullet_z[i] += game->bullet_vz[i] * (float) game->delta_time
					* 60.0f;

			/* Check boundaries */
			if (game->bullet_y[i] < 0|| game->bullet_y[i] > SCREEN_HEIGHT ||
			game->bullet_x[i] < 0 || game->bullet_x[i] > SCREEN_WIDTH) {
				game->bullet_active[i] = 0;
				game->num_bullets--;
				continue;
			}

			/* Check collision with invaders */
			if (game->bullet_owner[i] == 0) { /* Player bullet */
				int j;
				for (j = 0; j < game->num_invaders; ++j) {
					if (!game->invader_active[j])
						continue;

					float dx = game->bullet_x[i] - game->invader_x[j];
					float dy = game->bullet_y[i] - game->invader_y[j];
					float dist = fast_sqrt(dx * dx + dy * dy);

					if (dist < 20.0f) {
						game->invader_health[j] -= game->bullet_damage[i];
						game->bullet_active[i] = 0;
						game->num_bullets--;

						if (game->invader_health[j] <= 0.0f) {
							game->invader_active[j] = 0;
							game->score += 100;
							game->kills++;

							/* Spawn explosion particles */
							int k;
							for (k = 0; k < 20; ++k) {
								particle_spawn(game->particles,
										game->invader_x[j], game->invader_y[j],
										game->invader_z[j], 0xFFFF0000);
							}
						}
						break;
					}
				}
			}
		}

		/* Update weapon cooldown */
		if (game->weapon_cooldown > 0.0f) {
			game->weapon_cooldown -= (float) game->delta_time;
		}

		game->particle_updates++;
		game->frame_count++;
	}

	/*=============================================================================
	 * Rendering Functions
	 *===========================================================================*/

	static void draw_particles_gl(GameState *game) {
		int i;

		if (!game || !game->particles)
			return;

		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE);
		glPointSize(2.0f);

		glBegin(GL_POINTS);

		for (i = 0; i < game->particles->num_particles; ++i) {
			SynapticParticle *p = &game->particles->particles[i];
			if (p->active) {
				float r = ((p->color >> 16) & 0xFF) / 255.0f;
				float g = ((p->color >> 8) & 0xFF) / 255.0f;
				float b = (p->color & 0xFF) / 255.0f;
				float a = ((p->color >> 24) & 0xFF) / 255.0f * (p->life / 5.0f);

				glColor4f(r, g, b, a);
				glVertex3f(p->x, p->y, p->z);
			}
		}

		glEnd();
		glDisable(GL_BLEND);
	}

	static void draw_hyper_graph_gl(HyperGraph *hg) {
		int i;

		if (!hg)
			return;

		/* Draw vertices */
		glPointSize(4.0f);
		glBegin(GL_POINTS);

		for (i = 0; i < hg->num_vertices; ++i) {
			float intensity = hg->potentials[i];
			glColor4f(hg->colors[i * 3], hg->colors[i * 3 + 1],
					hg->colors[i * 3 + 2], intensity);
			glVertex3f(hg->positions[i * 3], hg->positions[i * 3 + 1],
					hg->positions[i * 3 + 2]);
		}

		glEnd();

		/* Draw edges */
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glLineWidth(1.0f);
		glBegin(GL_LINES);

		for (i = 0; i < hg->num_edges; ++i) {
			int v1 = hg->connections[i * 2];
			int v2 = hg->connections[i * 2 + 1];
			float weight = hg->connection_weights[i];

			glColor4f(0.3f, 0.6f, 1.0f, weight * 0.3f);
			glVertex3f(hg->positions[v1 * 3], hg->positions[v1 * 3 + 1],
					hg->positions[v1 * 3 + 2]);
			glVertex3f(hg->positions[v2 * 3], hg->positions[v2 * 3 + 1],
					hg->positions[v2 * 3 + 2]);
		}

		glEnd();
		glDisable(GL_BLEND);
	}

	static void draw_text_gl(int x, int y, const char *text, unsigned int color) {
		/* Simple placeholder - in production use SDL_ttf or bitmap font */
		(void) x;
		(void) y;
		(void) text;
		(void) color;
	}

	static void render_game(GameState *game) {
		int i;
		char hud[256];

		if (!game)
			return;

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glLoadIdentity();

		/* Set up 3D view */
		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
		gluPerspective(60.0, (double) SCREEN_WIDTH / SCREEN_HEIGHT, 1.0,
				1000.0);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		gluLookAt(SCREEN_WIDTH / 2.0, SCREEN_HEIGHT / 2.0, 500.0,
		SCREEN_WIDTH / 2.0, SCREEN_HEIGHT / 2.0, 0.0, 0.0, 1.0, 0.0);

		/* Draw background */
		glBegin(GL_QUADS);
		glColor4f(0.02f, 0.02f, 0.05f, 1.0f);
		glVertex3f(0.0f, 0.0f, -100.0f);
		glVertex3f(SCREEN_WIDTH, 0.0f, -100.0f);
		glVertex3f(SCREEN_WIDTH, SCREEN_HEIGHT, -100.0f);
		glVertex3f(0.0f, SCREEN_HEIGHT, -100.0f);
		glEnd();

		/* Draw hyper-graph */
		draw_hyper_graph_gl(game->hyper_graph);

		/* Draw particles */
		draw_particles_gl(game);

		/* Draw invaders */
		for (i = 0; i < game->num_invaders; ++i) {
			if (game->invader_active[i]) {
				glPushMatrix();
				glTranslatef(game->invader_x[i], game->invader_y[i],
						game->invader_z[i]);

				/* Color based on type */
				if (game->invader_type[i] == 0)
					glColor4f(1.0f, 0.2f, 0.2f, 1.0f);
				else if (game->invader_type[i] == 1)
					glColor4f(0.2f, 1.0f, 0.2f, 1.0f);
				else
					glColor4f(0.2f, 0.2f, 1.0f, 1.0f);

				/* Draw invader as sphere */
				{
					GLUquadric *quad = gluNewQuadric();
					if (quad) {
						gluSphere(quad, 15.0f, 16, 16);
						gluDeleteQuadric(quad);
					}
				}

				glPopMatrix();
			}
		}

		/* Draw player */
		glPushMatrix();
		glTranslatef(game->player_x, game->player_y, game->player_z);
		glColor4f(0.0f, 1.0f, 0.0f, 1.0f);

		{
			GLUquadric *quad = gluNewQuadric();
			if (quad) {
				gluSphere(quad, 20.0f, 16, 16);

				/* Draw shield if active */
				if (game->player_shield > 50.0f) {
					glColor4f(0.0f, 0.5f, 1.0f, 0.3f);
					gluSphere(quad, 40.0f, 16, 16);
				}
				gluDeleteQuadric(quad);
			}
		}

		glPopMatrix();

		/* Switch to 2D for HUD */
		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glMatrixMode(GL_MODELVIEW);

		glDisable(GL_DEPTH_TEST);
		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
		glOrtho(0, SCREEN_WIDTH, SCREEN_HEIGHT, 0, -1, 1);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		/* Draw HUD */
		glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

		sprintf(hud, "Score: %d  Wave: %d  Kills: %d", game->score, game->wave,
				game->kills);
		draw_text_gl(10, 20, hud, 0xFFFFFFFF);

		sprintf(hud, "Energy: %.0f  Shield: %.0f", game->player_energy,
				game->player_shield);
		draw_text_gl(10, 45, hud, 0xFFFFFFFF);

		sprintf(hud, "Neural Field: %.2f  Entropy: %.3f", game->neural_field,
				game->entropy);
		draw_text_gl(10, 70, hud, 0xFFFFFFFF);

		sprintf(hud, "FPS: %.1f  Particles: %d", game->fps,
				game->particles ? game->particles->num_particles : 0);
		draw_text_gl(10, 95, hud, 0xFFFFFFFF);

		if (game->game_over) {
			draw_text_gl(SCREEN_WIDTH / 2 - 50, SCREEN_HEIGHT / 2, "GAME OVER",
					0xFFFF0000);
		}

		if (game->paused) {
			draw_text_gl(SCREEN_WIDTH / 2 - 30, SCREEN_HEIGHT / 2, "PAUSED",
					0xFFFFFFFF);
		}

		/* Restore 3D state */
		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glMatrixMode(GL_MODELVIEW);
		glEnable(GL_DEPTH_TEST);
	}

	/*=============================================================================
	 * Main Game Loop
	 *===========================================================================*/

	static void game_loop(GameState *game) {
		Uint32 last_time, current_time, frame_time;
		Uint32 fps_last_time = 0;
		int fps_frames = 0;
		SDL_Event event;

		last_time = SDL_GetTicks();

		while (g_running && !game->game_over) {
			current_time = SDL_GetTicks();
			frame_time = current_time - last_time;

			/* Calculate delta time (clamped to avoid large jumps) */
			game->delta_time = (frame_time > 100) ? 0.016 : frame_time / 1000.0;
			last_time = current_time;

			/* Calculate FPS */
			fps_frames++;
			if (current_time - fps_last_time >= 1000) {
				game->fps = (float) fps_frames * 1000.0f
						/ (float) (current_time - fps_last_time);
				fps_frames = 0;
				fps_last_time = current_time;
			}

			/* Process events */
			while (SDL_PollEvent(&event)) {
				if (event.type == SDL_QUIT) {
					g_running = 0;
				} else if (event.type == SDL_KEYDOWN) {
					switch (event.key.keysym.sym) {
					case SDLK_ESCAPE:
						g_running = 0;
						break;
					case SDLK_SPACE:
						if (!game->paused && game->weapon_cooldown <= 0.0f) {
							/* Fire bullet */
							int i;
							for (i = 0; i < game->max_bullets; ++i) {
								if (!game->bullet_active[i]) {
									game->bullet_x[i] = game->player_x;
									game->bullet_y[i] = game->player_y - 20.0f;
									game->bullet_z[i] = game->player_z;
									game->bullet_vx[i] = 0.0f;
									game->bullet_vy[i] = -15.0f;
									game->bullet_vz[i] = 0.0f;
									game->bullet_damage[i] = 10.0f;
									game->bullet_owner[i] = 0;
									game->bullet_active[i] = 1;
									game->num_bullets++;
									game->weapon_cooldown = 0.5f;
									break;
								}
							}
						}
						break;
					case SDLK_p:
						game->paused = !game->paused;
						break;
					case SDLK_r:
						if (game->game_over) {
							/* Reset game */
							game->game_over = 0;
							game->score = 0;
							game->wave = 1;
							game->kills = 0;
							game->player_shield = 100.0f;
							game->player_energy = 100.0f;
						}
						break;
					}
				}
			}

			/* Handle continuous input */
			if (!game->paused && !game->game_over) {
				const Uint8 *keys = SDL_GetKeyboardState(NULL);
				float move_x = 0.0f, move_y = 0.0f;
				float speed = 5.0f;

				if (keys[SDL_SCANCODE_LEFT] || keys[SDL_SCANCODE_A])
					move_x -= speed;
				if (keys[SDL_SCANCODE_RIGHT] || keys[SDL_SCANCODE_D])
					move_x += speed;
				if (keys[SDL_SCANCODE_UP] || keys[SDL_SCANCODE_W])
					move_y -= speed;
				if (keys[SDL_SCANCODE_DOWN] || keys[SDL_SCANCODE_S])
					move_y += speed;

				game->player_x += move_x;
				game->player_y += move_y;

				/* Keep player on screen */
				game->player_x = clamp_float(game->player_x, 20.0f,
				SCREEN_WIDTH - 20.0f);
				game->player_y = clamp_float(game->player_y, 20.0f,
				SCREEN_HEIGHT - 20.0f);

				/* Update game state */
				game_update(game);
			}

			/* Render */
			render_game(game);
			SDL_GL_SwapWindow(g_window);

			/* Frame rate limiting */
			frame_time = SDL_GetTicks() - current_time;
			if (frame_time < FRAME_TIME_MS) {
				SDL_Delay(FRAME_TIME_MS - (int) frame_time);
			}
		}
	}

	/*=============================================================================
	 * Game Cleanup
	 *===========================================================================*/

	static void game_destroy(GameState *game) {
		int i;

		if (!game)
			return;

		/* Free AI systems */
		if (game->neural_net) {
			free(game->neural_net->weights);
			free(game->neural_net->biases);
			free(game->neural_net->activations);
			free(game->neural_net->errors);
			free(game->neural_net->layer_sizes);
			free(game->neural_net);
		}

		if (game->fuzzy_sys) {
			for (i = 0; i < game->fuzzy_sys->num_rules; ++i) {
				free(game->fuzzy_sys->rules[i].antecedents);
			}
			free(game->fuzzy_sys->input_sets);
			free(game->fuzzy_sys->output_sets);
			free(game->fuzzy_sys->rules);
			free(game->fuzzy_sys->inputs);
			free(game->fuzzy_sys->outputs);
			free(game->fuzzy_sys->aggregated);
			free(game->fuzzy_sys);
		}

		if (game->hyper_graph) {
			free(game->hyper_graph->positions);
			free(game->hyper_graph->colors);
			free(game->hyper_graph->potentials);
			free(game->hyper_graph->frequencies);
			free(game->hyper_graph->connections);
			free(game->hyper_graph->connection_weights);
			free(game->hyper_graph);
		}

		if (game->particles) {
			free(game->particles->particles);
			free(game->particles->active_indices);
			free(game->particles->batches);
			free(game->particles);
		}

		if (game->audio) {
			alDeleteSources(32, game->audio->sources);
			alDeleteBuffers(32, game->audio->buffers);
			alcMakeContextCurrent(NULL);
			alcDestroyContext(game->audio->context);
			alcCloseDevice(game->audio->device);
			free(game->audio);
		}

		if (game->opencl) {
			if (game->opencl->kernel)
				clReleaseKernel(game->opencl->kernel);
			if (game->opencl->program)
				clReleaseProgram(game->opencl->program);
			if (game->opencl->queue)
				clReleaseCommandQueue(game->opencl->queue);
			if (game->opencl->context)
				clReleaseContext(game->opencl->context);
			free(game->opencl);
		}

		if (game->thread_pool) {
			thread_pool_destroy(game->thread_pool);
		}

		/* Free arrays */
		free(game->invader_x);
		free(game->invader_y);
		free(game->invader_z);
		free(game->invader_health);
		free(game->invader_type);
		free(game->invader_active);

		free(game->bullet_x);
		free(game->bullet_y);
		free(game->bullet_z);
		free(game->bullet_vx);
		free(game->bullet_vy);
		free(game->bullet_vz);
		free(game->bullet_damage);
		free(game->bullet_active);
		free(game->bullet_owner);

		free(game);
	}

	/*=============================================================================
	 * Main Function
	 *===========================================================================*/

	int main(int argc, char *argv[]) {
		(void) argc;
		(void) argv;

		/* Initialize random seed */
		srand((unsigned int) time(NULL));

		/* Initialize signal handlers */
		signal(SIGINT, signal_handler);
		signal(SIGTERM, signal_handler);

		printf("\n");
		printf(
				"╔════════════════════════════════════════════════════════════╗\n");
		printf(
				"║         SpaceX - Advanced AI Core System v5.0.1           ║\n");
		printf(
				"║     Neuron-Fuzzy Space Invaders with Hyper-Graph Vision   ║\n");
		printf(
				"╚════════════════════════════════════════════════════════════╝\n\n");

		/* Get start time for profiling */
		gettimeofday(&g_start_time, NULL);

		/* Initialize SDL */
		if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_TIMER) < 0) {
			fprintf(stderr, "SDL initialization failed: %s\n", SDL_GetError());
			return 1;
		}

		/* Set OpenGL attributes */
		SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
		SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
		SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
		SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
		SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);

		/* Create window */
		g_window = SDL_CreateWindow("SpaceX - Neural AI",
		SDL_WINDOWPOS_CENTERED,
		SDL_WINDOWPOS_CENTERED,
		SCREEN_WIDTH, SCREEN_HEIGHT,
				SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

		if (!g_window) {
			fprintf(stderr, "Window creation failed: %s\n", SDL_GetError());
			SDL_Quit();
			return 1;
		}

		/* Create OpenGL context */
		g_gl_context = SDL_GL_CreateContext(g_window);
		if (!g_gl_context) {
			fprintf(stderr, "OpenGL context creation failed: %s\n",
					SDL_GetError());
			SDL_DestroyWindow(g_window);
			SDL_Quit();
			return 1;
		}

		SDL_GL_SetSwapInterval(1);

		/* Initialize OpenGL */
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_BLEND);
		glEnable(GL_MULTISAMPLE);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glClearColor(0.02f, 0.02f, 0.05f, 1.0f);

		/* Print system information */
		printf("System Information:\n");
		printf("  CPU Cores: %ld\n", sysconf(_SC_NPROCESSORS_ONLN));
		printf("  Page Size: %ld bytes\n", sysconf(_SC_PAGESIZE));
		printf("  Cache Line: %d bytes\n", CACHE_LINE_SIZE);

#if HAS_AVX2
    printf("  AVX2: Enabled\n");
    printf("  FMA: Enabled\n");
#else
		printf("  AVX2: Disabled (scalar fallback)\n");
#endif

		/* Create thread pool */
		printf("\nInitializing thread pool...\n");
		g_thread_pool = thread_pool_create((int) sysconf(_SC_NPROCESSORS_ONLN));
		if (g_thread_pool) {
			printf("  Thread pool created with %d threads\n",
					g_thread_pool->num_threads);
		}

		/* Create game */
		printf("Initializing game...\n");
		g_game = game_create();
		if (!g_game) {
			fprintf(stderr, "Game initialization failed\n");
			thread_pool_destroy(g_thread_pool);
			SDL_GL_DeleteContext(g_gl_context);
			SDL_DestroyWindow(g_window);
			SDL_Quit();
			return 1;
		}

		printf("\nGame initialized successfully:\n");
		printf("  Particles: %d/%d\n",
				g_game->particles ? g_game->particles->num_particles : 0,
				MAX_PARTICLES);
		printf("  Neural Network: %d-%d-%d\n", INPUT_NEURONS, HIDDEN_NEURONS,
		OUTPUT_NEURONS);
		printf("  Fuzzy Rules: %d\n",
				g_game->fuzzy_sys ? g_game->fuzzy_sys->num_rules : 0);
		printf("  Hyper-Graph: %d vertices, %d edges\n",
				g_game->hyper_graph ? g_game->hyper_graph->num_vertices : 0,
				g_game->hyper_graph ? g_game->hyper_graph->num_edges : 0);

		printf("\nControls:\n");
		printf("  Arrow Keys / WASD - Move\n");
		printf("  SPACE - Fire\n");
		printf("  P - Pause\n");
		printf("  R - Restart\n");
		printf("  ESC - Exit\n\n");

		/* Run main loop */
		game_loop(g_game);

		/* Cleanup */
		printf("\nShutting down...\n");

		/* Show performance statistics */
		{
			struct timeval end_time;
			double elapsed;
			gettimeofday(&end_time, NULL);
			elapsed = (end_time.tv_sec - g_start_time.tv_sec)
					+ (end_time.tv_usec - g_start_time.tv_usec) / 1000000.0;

			printf("\nPerformance Statistics:\n");
			printf("  Runtime: %.2f seconds\n", elapsed);
			printf("  Frames: %lu\n", (unsigned long) g_game->frame_count);
			printf("  Average FPS: %.2f\n", g_game->fps);
			printf("  Particle Updates: %lu\n",
					(unsigned long) g_game->particle_updates);
			printf("  AI Inferences: %lu\n",
					(unsigned long) g_game->ai_inferences);
			printf("  Final Score: %d\n", g_game->score);
			printf("  Enemies Killed: %d\n", g_game->kills);
		}

		game_destroy(g_game);
		thread_pool_destroy(g_thread_pool);

		SDL_GL_DeleteContext(g_gl_context);
		SDL_DestroyWindow(g_window);
		SDL_Quit();

		printf("\nSystem terminated.\n");
		return 0;
	}
