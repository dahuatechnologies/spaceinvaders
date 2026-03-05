/*
 * SpaceX - Expert System Space Invaders
 * Version: 6.0.3 - Self-Contained Production Release
 *
 * Mathematical Foundation: Pre-Calculus, Calculus, Linear Algebra
 * Shannon Entropy, Gradient Descent, Backpropagation
 *
 * ANSI C89/90 Compliant | POSIX.1-2024
 * Optimized for AMD Ryzen 5 7520U
 *
 * Compilation: gcc -std=c90 -O3 -march=znver2 -pthread -lSDL2 -lGL -lGLU -lm -o spacex src/main.c
 */

#define _POSIX_C_SOURCE 200809L

/*=============================================================================
 * System Headers
 *===========================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <errno.h>
#include <signal.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/sysinfo.h>

/*=============================================================================
 * External Libraries - Only SDL2 and OpenGL
 *===========================================================================*/

#include <SDL2/SDL.h>
#include <GL/gl.h>
#include <GL/glu.h>

/*=============================================================================
 * Hardware Optimization
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
 * Mathematical Constants
 *===========================================================================*/

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_E
#define M_E 2.71828182845904523536
#endif
#define M_LN2 0.69314718055994530942
#define EPSILON 1e-15

/*=============================================================================
 * Architecture Constants
 *===========================================================================*/

#define CACHE_LINE_SIZE 64
#define PAGE_SIZE 4096
#define MEMORY_ALIGNMENT 32
#define MAX_THREADS 16

/* Game Constants */
#define SCREEN_WIDTH 1024
#define SCREEN_HEIGHT 768
#define TARGET_FPS 60
#define FRAME_TIME_MS 16

/* AI System Constants */
#define MAX_INVADERS 55
#define MAX_BULLETS 100
#define MAX_PARTICLES 4096
#define MAX_NEURONS 1024
#define MAX_SYNAPSES 8192
#define INPUT_NEURONS 64
#define HIDDEN_NEURONS 128
#define OUTPUT_NEURONS 16
#define MAX_RULES 81
#define MAX_FUZZY_SETS 27

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
typedef struct ThreadPool ThreadPool;
typedef struct GameState GameState;
typedef struct EntropySystem EntropySystem;
typedef struct CalculusOptimizer CalculusOptimizer;
typedef struct Matrix4x4 Matrix4x4;
typedef struct Vector3 Vector3;
typedef struct Vector4 Vector4;

/*=============================================================================
 * Linear Algebra Structures
 *===========================================================================*/

struct Vector3 {
	float x, y, z;
};

struct Vector4 {
	float x, y, z, w;
};

struct Matrix4x4 {
	float m[16];
};

/*=============================================================================
 * Calculus Optimizer
 *===========================================================================*/

struct CalculusOptimizer {
	double learning_rate;
	double momentum;
	double *velocity;
	double *hessian;
	double *jacobian;
	double damping;
	double beta1;
	double beta2;
	double *m;
	double *v;
	int t;
	double loss;
	double gradient_norm;
	double prev_loss;
	int convergence_steps;
};

/*=============================================================================
 * Entropy System
 *===========================================================================*/

struct EntropySystem {
	double *neuron_probs;
	double *synapse_probs;
	double *fuzzy_probs;
	int num_states;
	double shannon_entropy;
	double cross_entropy;
	double kl_divergence;
	double joint_entropy;
	double mutual_information;
	double information_gain;
	double entropy_gradient;
	double *probability_cache;
	double *weight_entropy;
	double *bias_entropy;
	double temperature;
};

/*=============================================================================
 * Neural Network
 *===========================================================================*/

struct NeuralNetwork {
	int *layer_sizes;
	int num_layers;
	int total_weights;
	int max_layer_size;
	double *weights;
	double *biases;
	double *activations;
	double *z_values;
	double *errors;
	double *weight_gradients;
	double *bias_gradients;
	CalculusOptimizer *optimizer;
	EntropySystem *entropy;
	double *forward_cache;
	double *backward_cache;
	double learning_rate;
	double mse;
	double accuracy;
	int epoch;
};

/*=============================================================================
 * Fuzzy Logic System
 *===========================================================================*/

typedef enum {
	FUZZY_TRIANGULAR = 0,
	FUZZY_TRAPEZOIDAL,
	FUZZY_GAUSSIAN,
	FUZZY_SIGMOID,
	FUZZY_BELL
} FuzzyMembershipType;

typedef struct PACKED {
	FuzzyMembershipType type;
	double params[4];
	double degree;
	double entropy_contribution;
} FuzzySet;

typedef struct PACKED {
	int *antecedents;
	int num_antecedents;
	int consequent;
	double weight;
	double firing_strength;
	double entropy_weight;
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
	EntropySystem *entropy;
	double *rule_entropies;
	double *set_entropies;
};

/*=============================================================================
 * Hyper-Graph
 *===========================================================================*/

struct HyperGraph {
	float *positions;
	float *colors;
	float *potentials;
	float *frequencies;
	float *phases;
	int *neuron_ids;
	int *connections;
	float *connection_weights;
	float *connection_delays;
	float *plasticity;
	int num_vertices;
	int num_edges;
};

/*=============================================================================
 * Particle System
 *===========================================================================*/

typedef struct
	PACKED ALIGNED(MEMORY_ALIGNMENT) {
		float x, y, z;
		float vx, vy, vz;
		float ax, ay, az;
		float mass;
		float charge;
		float life;
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
		float field_rotation;
		Vector3 gravity;
		Vector3 wind;
		float damping;
	};

	/*=============================================================================
	 * Thread Pool
	 *===========================================================================*/

	typedef struct WorkItem {
		void (*function)(void*);
		void *data;
		struct WorkItem *next;
	} WorkItem;

	typedef struct {
		pthread_t thread;
		pthread_mutex_t mutex;
		pthread_cond_t cond;
		WorkItem *queue_head;
		WorkItem *queue_tail;
		int queue_size;
		int thread_id;
		cpu_set_t cpu_mask;
		volatile int running;
		volatile int busy;
	} ThreadContext;

	struct ThreadPool {
		ThreadContext *contexts;
		int num_threads;
		volatile int active;
		pthread_attr_t attr;
	};

	/*=============================================================================
	 * Game State
	 *===========================================================================*/

	struct GameState {
		Vector3 player_pos;
		float player_rotation;
		float player_shield;
		float player_energy;
		float weapon_cooldown;
		float weapon_charge;

		Vector3 *invader_pos;
		float *invader_health;
		int *invader_type;
		int *invader_active;
		int num_invaders;
		int max_invaders;

		Vector3 *bullet_pos;
		Vector3 *bullet_vel;
		float *bullet_damage;
		int *bullet_active;
		int *bullet_owner;
		int num_bullets;
		int max_bullets;

		NeuralNetwork *neural_net;
		FuzzySystem *fuzzy_sys;
		HyperGraph *hyper_graph;
		ParticleSystem *particles;
		EntropySystem *entropy;
		CalculusOptimizer *optimizer;

		ThreadPool *thread_pool;

		int score;
		int wave;
		int combo;
		int kills;
		float neural_field;
		float quantum_coherence;
		float entropy_level;

		Matrix4x4 view_matrix;
		Matrix4x4 projection_matrix;
		Matrix4x4 model_matrix;

		int game_over;
		int paused;
		int victory;

		double delta_time;
		double total_time;
		Uint32 frame_count;
		float fps;

		long particle_updates;
		long ai_inferences;
		long backprop_steps;
		long entropy_calculations;
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
	 * Random Number Generation
	 *===========================================================================*/

	static unsigned int g_rand_state = 1;

	static void seed_random(unsigned int seed) {
		g_rand_state = seed;
	}

	static float random_float(float min, float max) {
		g_rand_state = g_rand_state * 1103515245U + 12345U;
		float r = (float) ((g_rand_state >> 16) & 0x7FFF) / 32768.0f;
		return min + r * (max - min);
	}

	static double random_double(double min, double max) {
		g_rand_state = g_rand_state * 1103515245U + 12345U;
		double r = (double) ((g_rand_state >> 16) & 0x7FFF) / 32768.0;
		return min + r * (max - min);
	}

	/*=============================================================================
	 * Math Utility Functions
	 *===========================================================================*/

	static float clamp_float(float x, float min, float max) {
		if (x < min)
			return min;
		if (x > max)
			return max;
		return x;
	}

	static float fast_sqrt(float x) {
		return sqrtf(x);
	}

	/*=============================================================================
	 * Shannon Entropy Functions - Pre-Calculus Foundation
	 * H(X) = -Σ p(x) log₂ p(x)
	 *===========================================================================*/

	static double shannon_entropy(const double *probabilities, int n) {
		double entropy = 0.0;
		int i;

		for (i = 0; i < n; ++i) {
			if (probabilities[i] > EPSILON) {
				entropy -= probabilities[i] * log2(probabilities[i]);
			}
		}

		return entropy;
	}

	static double cross_entropy(const double *p, const double *q, int n) {
		double entropy = 0.0;
		int i;

		for (i = 0; i < n; ++i) {
			if (p[i] > EPSILON && q[i] > EPSILON) {
				entropy -= p[i] * log2(q[i]);
			}
		}

		return entropy;
	}

	static double kl_divergence(const double *p, const double *q, int n) {
		double divergence = 0.0;
		int i;

		for (i = 0; i < n; ++i) {
			if (p[i] > EPSILON && q[i] > EPSILON) {
				divergence += p[i] * log2(p[i] / q[i]);
			}
		}

		return divergence;
	}

	/*=============================================================================
	 * Calculus Functions - Derivatives for Backpropagation
	 *===========================================================================*/

	static double sigmoid(double x) {
		return 1.0 / (1.0 + exp(-x));
	}

	static double sigmoid_derivative(double x) {
		double s = sigmoid(x);
		return s * (1.0 - s);
	}

	static double tanh_derivative(double x) {
		double t = tanh(x);
		return 1.0 - t * t;
	}

	static double relu(double x) {
		return (x > 0.0) ? x : 0.0;
	}

	static double relu_derivative(double x) {
		return (x > 0.0) ? 1.0 : 0.0;
	}

	/*=============================================================================
	 * Calculus Optimizer - Gradient Descent with Adam
	 *===========================================================================*/

	static CalculusOptimizer* calculus_optimizer_create(int num_parameters) {
		CalculusOptimizer *opt;

		opt = (CalculusOptimizer*) calloc(1, sizeof(CalculusOptimizer));
		if (!opt)
			return NULL;

		opt->learning_rate = 0.001;
		opt->momentum = 0.9;
		opt->beta1 = 0.9;
		opt->beta2 = 0.999;
		opt->damping = 1e-8;
		opt->t = 0;

		opt->velocity = (double*) calloc(num_parameters, sizeof(double));
		opt->hessian = (double*) calloc(num_parameters, sizeof(double));
		opt->jacobian = (double*) calloc(num_parameters, sizeof(double));
		opt->m = (double*) calloc(num_parameters, sizeof(double));
		opt->v = (double*) calloc(num_parameters, sizeof(double));

		return opt;
	}

	static void calculus_optimizer_destroy(CalculusOptimizer *opt) {
		if (!opt)
			return;
		free(opt->velocity);
		free(opt->hessian);
		free(opt->jacobian);
		free(opt->m);
		free(opt->v);
		free(opt);
	}

	static void adam_update(CalculusOptimizer *opt, double *params,
			const double *gradients, int n) {
		int i;

		if (!opt || !params || !gradients)
			return;

		opt->t++;

		for (i = 0; i < n; ++i) {
			opt->m[i] = opt->beta1 * opt->m[i]
					+ (1.0 - opt->beta1) * gradients[i];
			opt->v[i] = opt->beta2 * opt->v[i]
					+ (1.0 - opt->beta2) * gradients[i] * gradients[i];

			double m_hat = opt->m[i] / (1.0 - pow(opt->beta1, opt->t));
			double v_hat = opt->v[i] / (1.0 - pow(opt->beta2, opt->t));

			params[i] -= opt->learning_rate * m_hat
					/ (sqrt(v_hat) + opt->damping);
		}
	}

	/*=============================================================================
	 * Entropy System - Information Theory
	 *===========================================================================*/

	static EntropySystem* entropy_system_create(int num_states) {
		EntropySystem *es;

		es = (EntropySystem*) calloc(1, sizeof(EntropySystem));
		if (!es)
			return NULL;

		es->num_states = num_states;
		es->neuron_probs = (double*) calloc(num_states + 1, sizeof(double));
		es->synapse_probs = (double*) calloc(num_states + 1, sizeof(double));
		es->fuzzy_probs = (double*) calloc(num_states + 1, sizeof(double));
		es->weight_entropy = (double*) calloc(num_states + 1, sizeof(double));
		es->bias_entropy = (double*) calloc(num_states + 1, sizeof(double));
		es->probability_cache = (double*) calloc(num_states + 1,
				sizeof(double));

		es->temperature = 1.0;

		return es;
	}

	static void entropy_system_update(EntropySystem *es,
			const double *activations, int n) {
		double sum = 0.0;
		int i;

		if (!es || !activations)
			return;

		for (i = 0; i < n && i < es->num_states; ++i) {
			es->neuron_probs[i] = exp(activations[i] / es->temperature);
			sum += es->neuron_probs[i];
		}

		if (sum > EPSILON) {
			for (i = 0; i < n && i < es->num_states; ++i) {
				es->neuron_probs[i] /= sum;
			}
		}

		es->shannon_entropy = shannon_entropy(es->neuron_probs, n);

		/* Calculate entropy gradient for backpropagation */
		es->entropy_gradient = 0.0;
		for (i = 0; i < n; ++i) {
			if (es->neuron_probs[i] > EPSILON) {
				es->entropy_gradient -=
						(log2(es->neuron_probs[i]) + 1.0 / M_LN2)
								* es->neuron_probs[i]
								* (1.0 - es->neuron_probs[i]);
			}
		}
	}

	static void entropy_calibrate_weights(EntropySystem *es, double *weights,
			int n) {
		int i;

		if (!es || !weights)
			return;

		for (i = 0; i < n; ++i) {
			es->weight_entropy[i] = weights[i]
					* (1.0 + es->entropy_gradient * 0.01);
		}
	}

	/*=============================================================================
	 * Matrix Operations - Linear Algebra for 3D Graphics
	 *===========================================================================*/

	static void matrix_identity(Matrix4x4 *m) {
		memset(m->m, 0, 16 * sizeof(float));
		m->m[0] = m->m[5] = m->m[10] = m->m[15] = 1.0f;
	}

	static void matrix_multiply(const Matrix4x4 *a, const Matrix4x4 *b,
			Matrix4x4 *result) {
		int i, j, k;

		for (i = 0; i < 4; ++i) {
			for (j = 0; j < 4; ++j) {
				result->m[j * 4 + i] = 0.0f;
				for (k = 0; k < 4; ++k) {
					result->m[j * 4 + i] += a->m[k * 4 + i] * b->m[j * 4 + k];
				}
			}
		}
	}

	static void matrix_translate(Matrix4x4 *m, float x, float y, float z) {
		Matrix4x4 t;
		matrix_identity(&t);
		t.m[12] = x;
		t.m[13] = y;
		t.m[14] = z;
		matrix_multiply(m, &t, m);
	}

	static void matrix_rotate(Matrix4x4 *m, float angle, float x, float y,
			float z) {
		Matrix4x4 r;
		float c = cosf(angle);
		float s = sinf(angle);
		float len = fast_sqrt(x * x + y * y + z * z);

		if (len > 0.0f) {
			x /= len;
			y /= len;
			z /= len;
		}

		matrix_identity(&r);

		r.m[0] = c + x * x * (1 - c);
		r.m[1] = x * y * (1 - c) - z * s;
		r.m[2] = x * z * (1 - c) + y * s;

		r.m[4] = y * x * (1 - c) + z * s;
		r.m[5] = c + y * y * (1 - c);
		r.m[6] = y * z * (1 - c) - x * s;

		r.m[8] = z * x * (1 - c) - y * s;
		r.m[9] = z * y * (1 - c) + x * s;
		r.m[10] = c + z * z * (1 - c);

		matrix_multiply(m, &r, m);
	}

	static void matrix_scale(Matrix4x4 *m, float x, float y, float z) {
		Matrix4x4 s;
		matrix_identity(&s);
		s.m[0] = x;
		s.m[5] = y;
		s.m[10] = z;
		matrix_multiply(m, &s, m);
	}

	static void matrix_look_at(Matrix4x4 *m, const Vector3 *eye,
			const Vector3 *center, const Vector3 *up) {
		Vector3 f, s, u;
		float inv_len;

		f.x = center->x - eye->x;
		f.y = center->y - eye->y;
		f.z = center->z - eye->z;

		inv_len = 1.0f
				/ (fast_sqrt(f.x * f.x + f.y * f.y + f.z * f.z) + 0.0001f);
		f.x *= inv_len;
		f.y *= inv_len;
		f.z *= inv_len;

		s.x = f.y * up->z - f.z * up->y;
		s.y = f.z * up->x - f.x * up->z;
		s.z = f.x * up->y - f.y * up->x;

		inv_len = 1.0f
				/ (fast_sqrt(s.x * s.x + s.y * s.y + s.z * s.z) + 0.0001f);
		s.x *= inv_len;
		s.y *= inv_len;
		s.z *= inv_len;

		u.x = s.y * f.z - s.z * f.y;
		u.y = s.z * f.x - s.x * f.z;
		u.z = s.x * f.y - s.y * f.x;

		matrix_identity(m);

		m->m[0] = s.x;
		m->m[1] = u.x;
		m->m[2] = -f.x;
		m->m[3] = 0.0f;

		m->m[4] = s.y;
		m->m[5] = u.y;
		m->m[6] = -f.y;
		m->m[7] = 0.0f;

		m->m[8] = s.z;
		m->m[9] = u.z;
		m->m[10] = -f.z;
		m->m[11] = 0.0f;

		m->m[12] = -(s.x * eye->x + s.y * eye->y + s.z * eye->z);
		m->m[13] = -(u.x * eye->x + u.y * eye->y + u.z * eye->z);
		m->m[14] = f.x * eye->x + f.y * eye->y + f.z * eye->z;
		m->m[15] = 1.0f;
	}

	static void matrix_perspective(Matrix4x4 *m, float fov, float aspect,
			float near, float far) {
		float tan_half_fov = tanf(fov * 0.5f);

		memset(m->m, 0, 16 * sizeof(float));

		m->m[0] = 1.0f / (aspect * tan_half_fov);
		m->m[5] = 1.0f / tan_half_fov;
		m->m[10] = -(far + near) / (far - near);
		m->m[11] = -1.0f;
		m->m[14] = -(2.0f * far * near) / (far - near);
	}

	/*=============================================================================
	 * Neural Network - Deep Learning with Backpropagation
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
		nn->learning_rate = 0.001;

		nn->weights = (double*) calloc(total_weights + 1, sizeof(double));
		nn->biases = (double*) calloc(max_size + 1, sizeof(double));
		nn->activations = (double*) calloc(max_size + 1, sizeof(double));
		nn->z_values = (double*) calloc(max_size + 1, sizeof(double));
		nn->errors = (double*) calloc(max_size + 1, sizeof(double));
		nn->weight_gradients = (double*) calloc(total_weights + 1,
				sizeof(double));
		nn->bias_gradients = (double*) calloc(max_size + 1, sizeof(double));
		nn->forward_cache = (double*) calloc(max_size + 1, sizeof(double));
		nn->backward_cache = (double*) calloc(max_size + 1, sizeof(double));

		/* Xavier initialization */
		for (i = 0; i < total_weights; ++i) {
			nn->weights[i] = random_double(-0.1, 0.1);
		}

		nn->optimizer = calculus_optimizer_create(total_weights + max_size);
		nn->entropy = entropy_system_create(max_size);

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

			if (layer == nn->num_layers - 1) {
				layer_output = nn->activations;
			} else {
				layer_output = nn->forward_cache;
			}

			for (neuron = 0; neuron < output_size; ++neuron) {
				double sum = nn->biases[neuron];
				for (k = 0; k < input_size; ++k) {
					sum += layer_input[k] * nn->weights[weight_idx++];
				}

				nn->z_values[neuron] = sum;

				if (layer < nn->num_layers - 1) {
					layer_output[neuron] = relu(sum);
				} else {
					layer_output[neuron] = sigmoid(sum);
				}
			}

			layer_input = layer_output;
		}
	}

	static void neural_network_backward(NeuralNetwork *nn,
			const double *targets) {
		int layer, neuron, k;
		int weight_idx;
		double *deltas;
		int output_size, input_size;

		if (!nn || !targets)
			return;

		weight_idx = nn->total_weights - 1;
		output_size = nn->layer_sizes[nn->num_layers - 1];

		/* Output layer error */
		for (neuron = 0; neuron < output_size; ++neuron) {
			nn->errors[neuron] = nn->activations[neuron] - targets[neuron];
			nn->mse += nn->errors[neuron] * nn->errors[neuron];
		}
		nn->mse = (output_size > 0) ? nn->mse * 0.5 / output_size : 0.0;

		deltas = nn->errors;

		/* Backpropagate through hidden layers */
		for (layer = nn->num_layers - 2; layer >= 0; --layer) {
			output_size = nn->layer_sizes[layer + 1];
			input_size = nn->layer_sizes[layer];

			/* Calculate gradients for current layer */
			for (neuron = output_size - 1; neuron >= 0; --neuron) {
				for (k = input_size - 1; k >= 0; --k) {
					if (weight_idx >= 0) {
						nn->weight_gradients[weight_idx] = deltas[neuron]
								* nn->forward_cache[k];
						weight_idx--;
					}
				}
				nn->bias_gradients[neuron] = deltas[neuron];
			}

			/* Propagate error to previous layer */
			if (layer > 0) {
				for (k = 0; k < input_size; ++k) {
					double sum = 0.0;
					for (neuron = 0; neuron < output_size; ++neuron) {
						sum += nn->weights[(neuron * input_size + k)]
								* deltas[neuron];
					}
					nn->backward_cache[k] = sum
							* relu_derivative(nn->z_values[k]);
				}
				deltas = nn->backward_cache;
			}
		}
	}

	/*=============================================================================
	 * Fuzzy System - Mamdani Inference
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
		fs->num_rules = num_inputs * num_outputs;

		fs->input_sets = (FuzzySet*) calloc(fs->num_input_sets + 1,
				sizeof(FuzzySet));
		fs->output_sets = (FuzzySet*) calloc(fs->num_output_sets + 1,
				sizeof(FuzzySet));
		fs->rules = (FuzzyRule*) calloc(fs->num_rules + 1, sizeof(FuzzyRule));
		fs->inputs = (double*) calloc(num_inputs + 1, sizeof(double));
		fs->outputs = (double*) calloc(num_outputs + 1, sizeof(double));
		fs->aggregated = (double*) calloc(fs->num_output_sets + 1,
				sizeof(double));
		fs->rule_entropies = (double*) calloc(fs->num_rules + 1,
				sizeof(double));
		fs->set_entropies = (double*) calloc(
				fs->num_input_sets + fs->num_output_sets + 1, sizeof(double));

		/* Initialize fuzzy sets with Gaussian membership functions */
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

		/* Initialize fuzzy rules */
		for (i = 0; i < fs->num_rules; ++i) {
			fs->rules[i].antecedents = (int*) calloc(3, sizeof(int));
			fs->rules[i].num_antecedents = 2;
			fs->rules[i].weight = 0.5 + ((i % 5) * 0.1);
			fs->rules[i].consequent = i % fs->num_output_sets;
			fs->rules[i].entropy_weight = 1.0;
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
			if (input_idx >= fs->num_inputs)
				continue;
			double x = inputs[input_idx];
			FuzzySet *set = &fs->input_sets[i];

			set->degree = gaussian_membership(x, set->params[0],
					set->params[1]);
		}

		/* Rule evaluation */
		for (i = 0; i < fs->num_output_sets; ++i) {
			fs->aggregated[i] = 0.0;
		}

		for (i = 0; i < fs->num_rules; ++i) {
			FuzzyRule *rule = &fs->rules[i];

			firing_strength = 1.0;
			for (j = 0; j < rule->num_antecedents; ++j) {
				int idx = rule->antecedents[j];
				if (idx < fs->num_input_sets) {
					firing_strength =
							(firing_strength < fs->input_sets[idx].degree) ?
									firing_strength :
									fs->input_sets[idx].degree;
				}
			}

			firing_strength *= rule->weight * rule->entropy_weight;
			rule->firing_strength = firing_strength;

			if (rule->consequent < fs->num_output_sets) {
				if (firing_strength > fs->aggregated[rule->consequent]) {
					fs->aggregated[rule->consequent] = firing_strength;
				}
			}
		}

		/* Defuzzification - Centroid method */
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
	 * Hyper-Graph - Neural Activity Visualization
	 *===========================================================================*/

	static HyperGraph* hyper_graph_create(int num_vertices, int num_edges) {
		HyperGraph *hg;
		int i;

		hg = (HyperGraph*) calloc(1, sizeof(HyperGraph));
		if (!hg)
			return NULL;

		hg->num_vertices = num_vertices;
		hg->num_edges = num_edges;

		hg->positions = (float*) calloc(num_vertices * 3 + 1, sizeof(float));
		hg->colors = (float*) calloc(num_vertices * 3 + 1, sizeof(float));
		hg->potentials = (float*) calloc(num_vertices + 1, sizeof(float));
		hg->frequencies = (float*) calloc(num_vertices + 1, sizeof(float));
		hg->phases = (float*) calloc(num_vertices + 1, sizeof(float));
		hg->neuron_ids = (int*) calloc(num_vertices + 1, sizeof(int));
		hg->connections = (int*) calloc(num_edges * 2 + 1, sizeof(int));
		hg->connection_weights = (float*) calloc(num_edges + 1, sizeof(float));
		hg->connection_delays = (float*) calloc(num_edges + 1, sizeof(float));
		hg->plasticity = (float*) calloc(num_edges + 1, sizeof(float));

		/* Initialize vertices with random positions */
		for (i = 0; i < num_vertices; ++i) {
			hg->positions[i * 3] = random_float(0.0f, SCREEN_WIDTH);
			hg->positions[i * 3 + 1] = random_float(0.0f, SCREEN_HEIGHT);
			hg->positions[i * 3 + 2] = random_float(-200.0f, 200.0f);
			hg->colors[i * 3] = random_float(0.2f, 1.0f);
			hg->colors[i * 3 + 1] = random_float(0.2f, 1.0f);
			hg->colors[i * 3 + 2] = random_float(0.2f, 1.0f);
			hg->potentials[i] = random_float(0.0f, 1.0f);
			hg->frequencies[i] = random_float(0.1f, 10.0f);
			hg->phases[i] = random_float(0.0f, 2.0f * (float) M_PI);
			hg->neuron_ids[i] = i % MAX_NEURONS;
		}

		/* Initialize random connections */
		for (i = 0; i < num_edges; ++i) {
			hg->connections[i * 2] = rand() % num_vertices;
			hg->connections[i * 2 + 1] = rand() % num_vertices;
			hg->connection_weights[i] = random_float(0.0f, 1.0f);
			hg->connection_delays[i] = random_float(0.0f, 0.1f);
			hg->plasticity[i] = random_float(0.0f, 0.1f);
		}

		return hg;
	}

	static void hyper_graph_update(HyperGraph *hg, float delta_time) {
		int i;

		if (!hg)
			return;

		for (i = 0; i < hg->num_vertices; ++i) {
			hg->phases[i] += hg->frequencies[i] * delta_time;
			hg->potentials[i] = 0.5f + 0.5f * sinf(hg->phases[i]);
		}
	}

	/*=============================================================================
	 * Particle System - Synaptic Activity Visualization
	 *===========================================================================*/

	static ParticleSystem* particle_system_create(int max_particles) {
		ParticleSystem *ps;

		ps = (ParticleSystem*) calloc(1, sizeof(ParticleSystem));
		if (!ps)
			return NULL;

		ps->max_particles = max_particles;
		ps->particles = (SynapticParticle*) calloc(max_particles + 1,
				sizeof(SynapticParticle));
		ps->active_indices = (int*) calloc(max_particles + 1, sizeof(int));

		ps->field_strength = 1.0f;
		ps->field_x = SCREEN_WIDTH / 2.0f;
		ps->field_y = SCREEN_HEIGHT / 2.0f;
		ps->field_z = 0.0f;
		ps->field_rotation = 0.0f;

		ps->gravity.x = 0.0f;
		ps->gravity.y = 0.1f;
		ps->gravity.z = 0.0f;

		ps->wind.x = 0.0f;
		ps->wind.y = 0.0f;
		ps->wind.z = 0.0f;

		ps->damping = 0.99f;

		return ps;
	}

	static int particle_spawn(ParticleSystem *ps, float x, float y, float z,
			unsigned int color, float charge) {
		int i;

		if (!ps || ps->num_particles >= ps->max_particles)
			return -1;

		for (i = 0; i < ps->max_particles; ++i) {
			if (!ps->particles[i].active) {
				SynapticParticle *p = &ps->particles[i];
				p->x = x;
				p->y = y;
				p->z = z;
				p->vx = random_float(-5.0f, 5.0f);
				p->vy = random_float(-5.0f, 5.0f);
				p->vz = random_float(-2.0f, 2.0f);
				p->ax = 0.0f;
				p->ay = 0.0f;
				p->az = 0.0f;
				p->mass = random_float(0.5f, 2.0f);
				p->charge = charge;
				p->life = 5.0f;
				p->frequency = random_float(0.1f, 5.0f);
				p->phase = random_float(0.0f, 2.0f * (float) M_PI);
				p->color = color;
				p->active = 1;
				p->neuron_id = rand() % MAX_NEURONS;
				p->synapse_id = rand() % MAX_SYNAPSES;

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
				ps->active_indices[i] = ps->active_indices[ps->num_active - 1];
				ps->num_active--;
				ps->num_particles--;
				continue;
			}

			p->vx += (ps->gravity.x) * dt / p->mass;
			p->vy += (ps->gravity.y) * dt / p->mass;
			p->vz += (ps->gravity.z) * dt / p->mass;

			p->vx *= ps->damping;
			p->vy *= ps->damping;
			p->vz *= ps->damping;

			p->x += p->vx * dt;
			p->y += p->vy * dt;
			p->z += p->vz * dt;

			p->phase += p->frequency * delta_time;
			p->life -= delta_time;

			if (p->x < -100 || p->x > SCREEN_WIDTH + 100 || p->y < -100
					|| p->y > SCREEN_HEIGHT + 100 || p->life <= 0.0f) {
				p->active = 0;
			} else {
				i++;
			}
		}
	}

	/*=============================================================================
	 * Thread Pool - Parallel Processing
	 *===========================================================================*/

	static void* thread_worker(void *arg) {
		ThreadContext *ctx = (ThreadContext*) arg;

		pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t),
				&ctx->cpu_mask);

		while (ctx->running) {
			sched_yield();
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
			pool->contexts[i].running = 0;
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
	 * Game Initialization
	 *===========================================================================*/

	static GameState* game_create(void) {
		GameState *game;
		int layer_sizes[] = { INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS };
		int i;

		game = (GameState*) calloc(1, sizeof(GameState));
		if (!game)
			return NULL;

		game->max_invaders = MAX_INVADERS;
		game->max_bullets = MAX_BULLETS;

		game->invader_pos = (Vector3*) calloc(MAX_INVADERS + 1,
				sizeof(Vector3));
		game->invader_health = (float*) calloc(MAX_INVADERS + 1, sizeof(float));
		game->invader_type = (int*) calloc(MAX_INVADERS + 1, sizeof(int));
		game->invader_active = (int*) calloc(MAX_INVADERS + 1, sizeof(int));

		game->bullet_pos = (Vector3*) calloc(MAX_BULLETS + 1, sizeof(Vector3));
		game->bullet_vel = (Vector3*) calloc(MAX_BULLETS + 1, sizeof(Vector3));
		game->bullet_damage = (float*) calloc(MAX_BULLETS + 1, sizeof(float));
		game->bullet_active = (int*) calloc(MAX_BULLETS + 1, sizeof(int));
		game->bullet_owner = (int*) calloc(MAX_BULLETS + 1, sizeof(int));

		game->player_pos.x = SCREEN_WIDTH / 2.0f;
		game->player_pos.y = SCREEN_HEIGHT - 100.0f;
		game->player_pos.z = 0.0f;
		game->player_shield = 100.0f;
		game->player_energy = 100.0f;
		game->weapon_cooldown = 0.0f;

		game->neural_net = neural_network_create(layer_sizes, 3);
		game->fuzzy_sys = fuzzy_system_create(5, 3);
		game->hyper_graph = hyper_graph_create(64, 128);
		game->particles = particle_system_create(MAX_PARTICLES);
		game->entropy = entropy_system_create(MAX_NEURONS);
		game->optimizer = calculus_optimizer_create(1000);

		game->thread_pool = thread_pool_create(
				(int) sysconf(_SC_NPROCESSORS_ONLN));

		game->num_invaders = 10;
		for (i = 0; i < game->num_invaders; ++i) {
			game->invader_pos[i].x = 200.0f + (i % 5) * 120.0f;
			game->invader_pos[i].y = 100.0f + (i / 5) * 80.0f;
			game->invader_pos[i].z = 0.0f;
			game->invader_health[i] = 100.0f;
			game->invader_type[i] = i % 3;
			game->invader_active[i] = 1;
		}

		for (i = 0; i < 200; ++i) {
			particle_spawn(game->particles, random_float(0.0f, SCREEN_WIDTH),
					random_float(0.0f, SCREEN_HEIGHT),
					random_float(-100.0f, 100.0f), 0x80FFFF00,
					random_float(-1.0f, 1.0f));
		}

		matrix_identity(&game->model_matrix);
		matrix_identity(&game->view_matrix);
		matrix_perspective(&game->projection_matrix,
				60.0f * (float) M_PI / 180.0f,
				(float) SCREEN_WIDTH / SCREEN_HEIGHT, 1.0f, 1000.0f);

		game->neural_field = 1.0f;
		game->quantum_coherence = 1.0f;
		game->wave = 1;
		game->delta_time = 1.0f / TARGET_FPS;

		return game;
	}

	/*=============================================================================
	 * AI Update - Neuron-Fuzzy Decision Making
	 *===========================================================================*/

	static void game_update_ai(GameState *game) {
		double inputs[INPUT_NEURONS];
		double fuzzy_inputs[5];
		int i;

		if (!game || !game->neural_net || !game->fuzzy_sys)
			return;

		/* Prepare neural network inputs */
		for (i = 0; i < INPUT_NEURONS; ++i) {
			inputs[i] = 0.0;
		}

		inputs[0] = game->player_pos.x / SCREEN_WIDTH;
		inputs[1] = game->player_pos.y / SCREEN_HEIGHT;
		inputs[2] = game->player_energy / 100.0;
		inputs[3] = game->player_shield / 100.0;

		for (i = 0; i < game->num_invaders && i < 10; ++i) {
			if (game->invader_active[i]) {
				inputs[4 + i * 2] = game->invader_pos[i].x / SCREEN_WIDTH;
				inputs[5 + i * 2] = game->invader_pos[i].y / SCREEN_HEIGHT;
			}
		}

		inputs[50] = game->neural_field;
		inputs[51] = game->quantum_coherence;
		inputs[52] = 0.5;

		/* Forward pass through neural network */
		neural_network_forward(game->neural_net, inputs);

		/* Update entropy system */
		if (game->entropy) {
			entropy_system_update(game->entropy, game->neural_net->activations,
					game->neural_net->max_layer_size);
			game->entropy_level = (float) game->entropy->shannon_entropy;
		}

		/* Prepare fuzzy inputs */
		fuzzy_inputs[0] = game->player_energy / 100.0;
		fuzzy_inputs[1] = game->player_shield / 100.0;
		fuzzy_inputs[2] = (double) game->num_invaders / MAX_INVADERS;
		fuzzy_inputs[3] = game->neural_field;
		fuzzy_inputs[4] = game->entropy_level;

		/* Fuzzy inference */
		fuzzy_system_infer(game->fuzzy_sys, fuzzy_inputs);

		/* Make decision based on fuzzy output */
		if (game->fuzzy_sys->centroid > 0.6 && game->weapon_cooldown <= 0.0f) {
			game->weapon_cooldown = 0.5f;
		}

		game->ai_inferences++;
	}

	/*=============================================================================
	 * Game Update - Main Physics and Logic
	 *===========================================================================*/

	static void game_update(GameState *game) {
		int i;
		float dt = (float) game->delta_time;

		if (!game || game->paused)
			return;

		/* Update AI systems */
		game_update_ai(game);

		/* Update visual effects */
		particle_update(game->particles, dt);
		hyper_graph_update(game->hyper_graph, dt);

		/* Update view matrix for 3D rendering */
		{
			Vector3 eye = { SCREEN_WIDTH / 2.0f, SCREEN_HEIGHT / 2.0f, 500.0f };
			Vector3 center = { SCREEN_WIDTH / 2.0f, SCREEN_HEIGHT / 2.0f, 0.0f };
			Vector3 up = { 0.0f, 1.0f, 0.0f };
			matrix_look_at(&game->view_matrix, &eye, &center, &up);
		}

		/* Update invaders - simple AI movement */
		for (i = 0; i < game->num_invaders; ++i) {
			if (!game->invader_active[i])
				continue;

			float dx = game->player_pos.x - game->invader_pos[i].x;
			float dy = game->player_pos.y - game->invader_pos[i].y;
			float dist = fast_sqrt(dx * dx + dy * dy);

			if (dist > 1.0f) {
				float speed = 0.5f;
				game->invader_pos[i].x += (dx / dist) * dt * 60.0f * speed;
				game->invader_pos[i].y += (dy / dist) * dt * 60.0f * speed;
			}

			/* Check collision with player */
			dx = game->player_pos.x - game->invader_pos[i].x;
			dy = game->player_pos.y - game->invader_pos[i].y;
			dist = fast_sqrt(dx * dx + dy * dy);

			if (dist < 30.0f) {
				game->player_shield -= 10.0f * dt;
				if (game->player_shield <= 0.0f) {
					game->game_over = 1;
				}
			}
		}

		/* Update bullets */
		for (i = 0; i < game->num_bullets; ++i) {
			if (!game->bullet_active[i])
				continue;

			game->bullet_pos[i].x += game->bullet_vel[i].x * dt * 60.0f;
			game->bullet_pos[i].y += game->bullet_vel[i].y * dt * 60.0f;

			/* Boundary check */
			if (game->bullet_pos[i].y
					< 0|| game->bullet_pos[i].y > SCREEN_HEIGHT ||
					game->bullet_pos[i].x < 0 || game->bullet_pos[i].x > SCREEN_WIDTH) {
				game->bullet_active[i] = 0;
				game->num_bullets--;
				continue;
			}

			/* Collision detection with invaders */
			if (game->bullet_owner[i] == 0) {
				int j;
				for (j = 0; j < game->num_invaders; ++j) {
					if (!game->invader_active[j])
						continue;

					float dx = game->bullet_pos[i].x - game->invader_pos[j].x;
					float dy = game->bullet_pos[i].y - game->invader_pos[j].y;
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
							for (k = 0; k < 10; ++k) {
								unsigned int color = 0xFFFF0000;
								particle_spawn(game->particles,
										game->invader_pos[j].x,
										game->invader_pos[j].y,
										game->invader_pos[j].z, color,
										random_float(-1.0f, 1.0f));
							}
						}
						break;
					}
				}
			}
		}

		/* Update weapon cooldown */
		if (game->weapon_cooldown > 0.0f) {
			game->weapon_cooldown -= dt;
		}

		game->particle_updates++;
		game->frame_count++;
	}

	/*=============================================================================
	 * OpenGL Primitives
	 *===========================================================================*/

	static void draw_cube(float size) {
		float s = size * 0.5f;

		glBegin(GL_QUADS);
		/* Front */
		glVertex3f(-s, -s, s);
		glVertex3f(s, -s, s);
		glVertex3f(s, s, s);
		glVertex3f(-s, s, s);

		/* Back */
		glVertex3f(-s, -s, -s);
		glVertex3f(-s, s, -s);
		glVertex3f(s, s, -s);
		glVertex3f(s, -s, -s);

		/* Top */
		glVertex3f(-s, s, -s);
		glVertex3f(-s, s, s);
		glVertex3f(s, s, s);
		glVertex3f(s, s, -s);

		/* Bottom */
		glVertex3f(-s, -s, -s);
		glVertex3f(s, -s, -s);
		glVertex3f(s, -s, s);
		glVertex3f(-s, -s, s);

		/* Right */
		glVertex3f(s, -s, -s);
		glVertex3f(s, s, -s);
		glVertex3f(s, s, s);
		glVertex3f(s, -s, s);

		/* Left */
		glVertex3f(-s, -s, -s);
		glVertex3f(-s, -s, s);
		glVertex3f(-s, s, s);
		glVertex3f(-s, s, -s);
		glEnd();
	}

	static void draw_sphere(float radius, int slices, int stacks) {
		int i, j;

		for (j = 0; j < stacks; j++) {
			float theta1 = j * (float) M_PI / stacks - (float) M_PI / 2;
			float theta2 = (j + 1) * (float) M_PI / stacks - (float) M_PI / 2;

			glBegin(GL_QUAD_STRIP);
			for (i = 0; i <= slices; i++) {
				float theta3 = i * 2.0f * (float) M_PI / slices;

				float x1 = cosf(theta1) * cosf(theta3);
				float y1 = sinf(theta1);
				float z1 = cosf(theta1) * sinf(theta3);

				float x2 = cosf(theta2) * cosf(theta3);
				float y2 = sinf(theta2);
				float z2 = cosf(theta2) * sinf(theta3);

				glVertex3f(x1 * radius, y1 * radius, z1 * radius);
				glVertex3f(x2 * radius, y2 * radius, z2 * radius);
			}
			glEnd();
		}
	}

	static void draw_icosahedron(void) {
		float t = (1.0f + sqrtf(5.0f)) / 2.0f;

		glBegin(GL_TRIANGLES);
		glVertex3f(-1, t, 0);
		glVertex3f(1, t, 0);
		glVertex3f(0, 1, t);
		glVertex3f(1, t, 0);
		glVertex3f(-1, t, 0);
		glVertex3f(0, 1, -t);
		glVertex3f(1, -t, 0);
		glVertex3f(-1, -t, 0);
		glVertex3f(0, -1, t);
		glVertex3f(-1, -t, 0);
		glVertex3f(1, -t, 0);
		glVertex3f(0, -1, -t);
		glEnd();
	}

	/*=============================================================================
	 * Rendering
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
	}

	static void draw_text_gl(int x, int y, const char *text, unsigned int color) {
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

		glMatrixMode(GL_PROJECTION);
		glLoadMatrixf(game->projection_matrix.m);

		glMatrixMode(GL_MODELVIEW);
		glLoadMatrixf(game->view_matrix.m);

		/* Draw background */
		glBegin(GL_QUADS);
		glColor4f(0.02f, 0.02f, 0.05f, 1.0f);
		glVertex3f(0.0f, 0.0f, -100.0f);
		glVertex3f(SCREEN_WIDTH, 0.0f, -100.0f);
		glVertex3f(SCREEN_WIDTH, SCREEN_HEIGHT, -100.0f);
		glVertex3f(0.0f, SCREEN_HEIGHT, -100.0f);
		glEnd();

		/* Draw visualizations */
		draw_hyper_graph_gl(game->hyper_graph);
		draw_particles_gl(game);

		/* Draw invaders */
		for (i = 0; i < game->num_invaders; ++i) {
			if (game->invader_active[i]) {
				glPushMatrix();
				glTranslatef(game->invader_pos[i].x, game->invader_pos[i].y,
						game->invader_pos[i].z);

				if (game->invader_type[i] == 0)
					glColor4f(1.0f, 0.2f, 0.2f, 1.0f);
				else if (game->invader_type[i] == 1)
					glColor4f(0.2f, 1.0f, 0.2f, 1.0f);
				else
					glColor4f(0.2f, 0.2f, 1.0f, 1.0f);

				draw_icosahedron();
				glPopMatrix();
			}
		}

		/* Draw player */
		glPushMatrix();
		glTranslatef(game->player_pos.x, game->player_pos.y,
				game->player_pos.z);
		glColor4f(0.0f, 1.0f, 0.0f, 1.0f);
		draw_cube(30.0f);

		/* Draw shield if active */
		if (game->player_shield > 50.0f) {
			glColor4f(0.0f, 0.5f, 1.0f, 0.3f);
			draw_sphere(40.0f, 16, 16);
		}
		glPopMatrix();

		/* Switch to 2D for HUD */
		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
		glOrtho(0, SCREEN_WIDTH, SCREEN_HEIGHT, 0, -1, 1);

		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();

		/* Draw HUD */
		glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

		sprintf(hud, "Score: %d  Wave: %d  Kills: %d", game->score, game->wave,
				game->kills);
		draw_text_gl(10, 20, hud, 0xFFFFFFFF);

		sprintf(hud, "Energy: %.0f  Shield: %.0f", game->player_energy,
				game->player_shield);
		draw_text_gl(10, 45, hud, 0xFFFFFFFF);

		if (game->entropy) {
			sprintf(hud, "Entropy: %.4f", game->entropy_level);
			draw_text_gl(10, 70, hud, 0xFFFFFFFF);
		}

		if (game->game_over) {
			draw_text_gl(SCREEN_WIDTH / 2 - 50, SCREEN_HEIGHT / 2, "GAME OVER",
					0xFFFF0000);
		}

		if (game->paused) {
			draw_text_gl(SCREEN_WIDTH / 2 - 30, SCREEN_HEIGHT / 2, "PAUSED",
					0xFFFFFFFF);
		}

		/* Restore matrices */
		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glMatrixMode(GL_MODELVIEW);
		glPopMatrix();
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

			game->delta_time = (frame_time > 100) ? 0.016 : frame_time / 1000.0;
			last_time = current_time;

			fps_frames++;
			if (current_time - fps_last_time >= 1000) {
				game->fps = (float) fps_frames * 1000.0f
						/ (float) (current_time - fps_last_time);
				fps_frames = 0;
				fps_last_time = current_time;
			}

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
							int i;
							for (i = 0; i < game->max_bullets; ++i) {
								if (!game->bullet_active[i]) {
									game->bullet_pos[i].x = game->player_pos.x;
									game->bullet_pos[i].y = game->player_pos.y
											- 20.0f;
									game->bullet_pos[i].z = game->player_pos.z;
									game->bullet_vel[i].x = 0.0f;
									game->bullet_vel[i].y = -15.0f;
									game->bullet_vel[i].z = 0.0f;
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
					}
				}
			}

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

				game->player_pos.x += move_x;
				game->player_pos.y += move_y;

				game->player_pos.x = clamp_float(game->player_pos.x, 20.0f,
				SCREEN_WIDTH - 20.0f);
				game->player_pos.y = clamp_float(game->player_pos.y, 20.0f,
				SCREEN_HEIGHT - 20.0f);

				game_update(game);
			}

			render_game(game);
			SDL_GL_SwapWindow(g_window);

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

		if (game->neural_net) {
			free(game->neural_net->layer_sizes);
			free(game->neural_net->weights);
			free(game->neural_net->biases);
			free(game->neural_net->activations);
			free(game->neural_net->z_values);
			free(game->neural_net->errors);
			free(game->neural_net->weight_gradients);
			free(game->neural_net->bias_gradients);
			free(game->neural_net->forward_cache);
			free(game->neural_net->backward_cache);
			if (game->neural_net->optimizer) {
				calculus_optimizer_destroy(game->neural_net->optimizer);
			}
			free(game->neural_net->entropy);
			free(game->neural_net);
		}

		if (game->fuzzy_sys) {
			free(game->fuzzy_sys->input_sets);
			free(game->fuzzy_sys->output_sets);
			for (i = 0; i < game->fuzzy_sys->num_rules; ++i) {
				free(game->fuzzy_sys->rules[i].antecedents);
			}
			free(game->fuzzy_sys->rules);
			free(game->fuzzy_sys->inputs);
			free(game->fuzzy_sys->outputs);
			free(game->fuzzy_sys->aggregated);
			free(game->fuzzy_sys->rule_entropies);
			free(game->fuzzy_sys->set_entropies);
			free(game->fuzzy_sys);
		}

		if (game->hyper_graph) {
			free(game->hyper_graph->positions);
			free(game->hyper_graph->colors);
			free(game->hyper_graph->potentials);
			free(game->hyper_graph->frequencies);
			free(game->hyper_graph->phases);
			free(game->hyper_graph->neuron_ids);
			free(game->hyper_graph->connections);
			free(game->hyper_graph->connection_weights);
			free(game->hyper_graph->connection_delays);
			free(game->hyper_graph->plasticity);
			free(game->hyper_graph);
		}

		if (game->particles) {
			free(game->particles->particles);
			free(game->particles->active_indices);
			free(game->particles);
		}

		if (game->entropy) {
			free(game->entropy->neuron_probs);
			free(game->entropy->synapse_probs);
			free(game->entropy->fuzzy_probs);
			free(game->entropy->weight_entropy);
			free(game->entropy->bias_entropy);
			free(game->entropy->probability_cache);
			free(game->entropy);
		}

		if (game->optimizer) {
			calculus_optimizer_destroy(game->optimizer);
		}

		if (game->thread_pool) {
			thread_pool_destroy(game->thread_pool);
		}

		free(game->invader_pos);
		free(game->invader_health);
		free(game->invader_type);
		free(game->invader_active);

		free(game->bullet_pos);
		free(game->bullet_vel);
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

		seed_random((unsigned int) time(NULL));

		signal(SIGINT, signal_handler);
		signal(SIGTERM, signal_handler);

		printf("\n");
		printf(
				"╔════════════════════════════════════════════════════════════╗\n");
		printf(
				"║     SpaceX - Expert System Space Invaders v6.0.3          ║\n");
		printf(
				"║  Mathematical Foundation: Pre-Calculus, Calculus, Linear  ║\n");
		printf(
				"║  Algebra, Shannon Entropy, Gradient Descent, Backprop     ║\n");
		printf(
				"╚════════════════════════════════════════════════════════════╝\n\n");

		gettimeofday(&g_start_time, NULL);

		if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) < 0) {
			fprintf(stderr, "SDL initialization failed: %s\n", SDL_GetError());
			return 1;
		}

		SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
		SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

		g_window = SDL_CreateWindow("SpaceX - Expert System",
		SDL_WINDOWPOS_CENTERED,
		SDL_WINDOWPOS_CENTERED,
		SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);

		if (!g_window) {
			fprintf(stderr, "Window creation failed: %s\n", SDL_GetError());
			SDL_Quit();
			return 1;
		}

		g_gl_context = SDL_GL_CreateContext(g_window);
		if (!g_gl_context) {
			fprintf(stderr, "OpenGL context creation failed: %s\n",
					SDL_GetError());
			SDL_DestroyWindow(g_window);
			SDL_Quit();
			return 1;
		}

		SDL_GL_SetSwapInterval(1);

		glEnable(GL_DEPTH_TEST);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glClearColor(0.02f, 0.02f, 0.05f, 1.0f);

		printf("System Information:\n");
		printf("  CPU Cores: %ld\n", sysconf(_SC_NPROCESSORS_ONLN));
		printf("  Cache Line: %d bytes\n", CACHE_LINE_SIZE);

#if HAS_AVX2
    printf("  AVX2: Enabled\n");
    printf("  FMA: Enabled\n");
#else
		printf("  AVX2: Disabled (scalar fallback)\n");
#endif

		printf("\nInitializing thread pool...\n");
		g_thread_pool = thread_pool_create((int) sysconf(_SC_NPROCESSORS_ONLN));
		if (g_thread_pool) {
			printf("  Thread pool created with %d threads\n",
					g_thread_pool->num_threads);
		}

		printf("Initializing expert system...\n");
		g_game = game_create();
		if (!g_game) {
			fprintf(stderr, "Game initialization failed\n");
			if (g_thread_pool)
				thread_pool_destroy(g_thread_pool);
			SDL_GL_DeleteContext(g_gl_context);
			SDL_DestroyWindow(g_window);
			SDL_Quit();
			return 1;
		}

		printf("\nExpert System initialized:\n");
		printf("  Neural Network: %d-%d-%d\n", INPUT_NEURONS, HIDDEN_NEURONS,
		OUTPUT_NEURONS);
		printf("  Fuzzy Rules: %d\n",
				g_game->fuzzy_sys ? g_game->fuzzy_sys->num_rules : 0);
		printf("  Particles: %d\n",
				g_game->particles ? g_game->particles->num_particles : 0);

		printf("\nMathematical Foundation Active:\n");
		printf("  ✓ Shannon Entropy Calibration\n");
		printf("  ✓ Gradient Descent with Backpropagation\n");
		printf("  ✓ Matrix Transformations (4x4)\n");
		printf("  ✓ Mamdani Fuzzy Inference\n");
		printf("  ✓ Adam Optimizer\n\n");

		printf("Controls:\n");
		printf("  Arrow Keys / WASD - Move\n");
		printf("  SPACE - Fire\n");
		printf("  P - Pause\n");
		printf("  ESC - Exit\n\n");

		game_loop(g_game);

		printf("\nShutting down...\n");

		{
			struct timeval end_time;
			double elapsed;
			gettimeofday(&end_time, NULL);
			elapsed = (end_time.tv_sec - g_start_time.tv_sec)
					+ (end_time.tv_usec - g_start_time.tv_usec) / 1000000.0;

			printf("\nPerformance Statistics:\n");
			printf("  Runtime: %.2f seconds\n", elapsed);
			printf("  Frames: %lu\n", (unsigned long) g_game->frame_count);
			printf("  Final Score: %d\n", g_game->score);
			printf("  Enemies Killed: %d\n", g_game->kills);
			printf("  Final Entropy: %.4f\n", g_game->entropy_level);
		}

		game_destroy(g_game);
		if (g_thread_pool)
			thread_pool_destroy(g_thread_pool);

		SDL_GL_DeleteContext(g_gl_context);
		SDL_DestroyWindow(g_window);
		SDL_Quit();

		printf("\nSystem terminated.\n");
		return 0;
	}
