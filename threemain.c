/*
 * SpaceX - Expert System
 * Version: 0.1 - Production Release - C90 Compliant
 *
 * ANSI C89/90 Compliant | POSIX.1-2024
 * Optimized for AMD Ryzen 5 7520U
 */

#define _POSIX_C_SOURCE 200809L

/*=============================================================================
 * System Headers - C89/90 Compliant
 *===========================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <signal.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <sys/time.h>
#include <sys/sysinfo.h>

/*=============================================================================
 * External Library Headers
 *===========================================================================*/

#include <SDL2/SDL.h>
#include <GL/gl.h>
#include <GL/glu.h>

/*=============================================================================
 * Architecture Constants
 *===========================================================================*/

#define SCREEN_WIDTH            1024
#define SCREEN_HEIGHT           768
#define TARGET_FPS              60
#define MAX_INVADERS            55
#define MAX_BULLETS             100
#define MAX_PARTICLES           4096
#define INPUT_NEURONS           64
#define HIDDEN_NEURONS          128
#define OUTPUT_NEURONS          16
#define CACHE_LINE_SIZE         64
#define MEMORY_ALIGNMENT        32

/*=============================================================================
 * Type Definitions
 *===========================================================================*/

typedef struct NeuralNetwork NeuralNetwork;
typedef struct FuzzySystem FuzzySystem;
typedef struct HyperGraph HyperGraph;
typedef struct SynapticParticle SynapticParticle;
typedef struct NeuralGameState NeuralGameState;
typedef struct ThreadPool ThreadPool;

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
	double learning_rate;
};

struct FuzzySystem {
	double *inputs;
	double *outputs;
	double *rules;
	int num_rules;
	int num_inputs;
	int num_outputs;
	double centroid;
};

/*=============================================================================
 * Hyper-Graph Structures
 *===========================================================================*/

struct HyperGraph {
	float *positions;
	float *colors;
	float *potentials;
	int *connections;
	int num_vertices;
	int num_edges;
};

/*=============================================================================
 * Particle System
 *===========================================================================*/

struct SynapticParticle {
	float x, y, z;
	float vx, vy, vz;
	float ax, ay, az;
	float life;
	unsigned int color;
	int active;
};

/*=============================================================================
 * Thread Pool
 *===========================================================================*/

struct ThreadPool {
	pthread_t *threads;
	int num_threads;
	volatile int running;
};

/*=============================================================================
 * Game State
 *===========================================================================*/

struct NeuralGameState {
	/* Player */
	float player_x, player_y, player_z;
	float player_shield;
	float player_energy;
	float weapon_cooldown;

	/* Invaders */
	float invader_x[MAX_INVADERS];
	float invader_y[MAX_INVADERS];
	float invader_health[MAX_INVADERS];
	int invader_type[MAX_INVADERS];
	int num_invaders;

	/* Bullets */
	float bullet_x[MAX_BULLETS];
	float bullet_y[MAX_BULLETS];
	float bullet_vx[MAX_BULLETS];
	float bullet_vy[MAX_BULLETS];
	int bullet_active[MAX_BULLETS];
	int bullet_owner[MAX_BULLETS];
	int num_bullets;

	/* Particles */
	SynapticParticle particles[MAX_PARTICLES];
	int num_particles;

	/* AI Systems */
	NeuralNetwork *neural_net;
	FuzzySystem *fuzzy_sys;
	HyperGraph *hyper_graph;

	/* Game State */
	int score;
	int wave;
	float neural_field;
	float entropy;
	int game_over;
	int paused;

	/* Timing */
	double delta_time;
	Uint32 frame_count;
};

/*=============================================================================
 * Global State
 *===========================================================================*/

static SDL_Window *g_window = NULL;
static SDL_GLContext g_gl_context = NULL;
static volatile sig_atomic_t g_running = 1;
static NeuralGameState *g_game = NULL;
static ThreadPool *g_thread_pool = NULL;

/*=============================================================================
 * Signal Handler
 *===========================================================================*/

static void signal_handler(int sig) {
	(void) sig;
	g_running = 0;
}

/*=============================================================================
 * Math Utilities
 *===========================================================================*/

static float random_float(float min, float max) {
	return min + ((float) rand() / RAND_MAX) * (max - min);
}

static double sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

static double clamp(double x, double min, double max) {
	if (x < min)
		return min;
	if (x > max)
		return max;
	return x;
}

/*=============================================================================
 * Neural Network Implementation
 *===========================================================================*/

static NeuralNetwork* neural_network_create(int *layer_sizes, int num_layers) {
	NeuralNetwork *nn;
	int i, total = 0;

	nn = (NeuralNetwork*) calloc(1, sizeof(NeuralNetwork));
	if (!nn)
		return NULL;

	nn->num_layers = num_layers;
	nn->layer_sizes = (int*) calloc(num_layers, sizeof(int));
	nn->learning_rate = 0.01;

	if (!nn->layer_sizes) {
		free(nn);
		return NULL;
	}

	for (i = 0; i < num_layers; i++) {
		nn->layer_sizes[i] = layer_sizes[i];
		if (i > 0) {
			total += layer_sizes[i] * layer_sizes[i - 1];
		}
	}

	nn->total_weights = total;
	nn->weights = (double*) calloc(total, sizeof(double));
	nn->biases = (double*) calloc(layer_sizes[num_layers - 1], sizeof(double));
	nn->activations = (double*) calloc(layer_sizes[num_layers - 1],
			sizeof(double));
	nn->errors = (double*) calloc(layer_sizes[num_layers - 1], sizeof(double));

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
	for (i = 0; i < total; i++) {
		nn->weights[i] = ((double) rand() / RAND_MAX) * 0.2 - 0.1;
	}

	return nn;
}

static void neural_network_forward(NeuralNetwork *nn, double *inputs) {
	int i, j, k;
	int weight_idx = 0;
	double *layer_input;
	double *layer_output;

	layer_input = inputs;

	for (i = 1; i < nn->num_layers; i++) {
		int input_size = nn->layer_sizes[i - 1];
		int output_size = nn->layer_sizes[i];

		for (j = 0; j < output_size; j++) {
			double sum = 0.0;
			for (k = 0; k < input_size; k++) {
				sum += layer_input[k] * nn->weights[weight_idx++];
			}
			if (i == nn->num_layers - 1) {
				nn->activations[j] = sum;
			}
			layer_input = &nn->activations[0];
		}
	}
}

/*=============================================================================
 * Fuzzy System Implementation
 *===========================================================================*/

static FuzzySystem* fuzzy_system_create(int num_inputs, int num_outputs) {
	FuzzySystem *fs;

	fs = (FuzzySystem*) calloc(1, sizeof(FuzzySystem));
	if (!fs)
		return NULL;

	fs->num_inputs = num_inputs;
	fs->num_outputs = num_outputs;
	fs->num_rules = num_inputs * num_outputs;

	fs->inputs = (double*) calloc(num_inputs, sizeof(double));
	fs->outputs = (double*) calloc(num_outputs, sizeof(double));
	fs->rules = (double*) calloc(fs->num_rules, sizeof(double));

	if (!fs->inputs || !fs->outputs || !fs->rules) {
		free(fs->inputs);
		free(fs->outputs);
		free(fs->rules);
		free(fs);
		return NULL;
	}

	/* Initialize rules */
	{
		int i;
		for (i = 0; i < fs->num_rules; i++) {
			fs->rules[i] = 0.5;
		}
	}

	return fs;
}

static void fuzzy_system_infer(FuzzySystem *fs, double *inputs) {
	int i;
	double sum = 0.0;
	double weight_sum = 0.0;

	for (i = 0; i < fs->num_inputs; i++) {
		fs->inputs[i] = inputs[i];
	}

	/* Simple centroid defuzzification */
	for (i = 0; i < fs->num_rules; i++) {
		double firing_strength = 1.0;
		int j;
		for (j = 0; j < fs->num_inputs; j++) {
			firing_strength *= inputs[j];
		}
		sum += firing_strength * fs->rules[i];
		weight_sum += firing_strength;
	}

	fs->centroid = (weight_sum > 0) ? sum / weight_sum : 0.0;
}

/*=============================================================================
 * Hyper-Graph Implementation
 *===========================================================================*/

static HyperGraph* hyper_graph_create(int num_vertices, int num_edges) {
	HyperGraph *hg;

	hg = (HyperGraph*) calloc(1, sizeof(HyperGraph));
	if (!hg)
		return NULL;

	hg->num_vertices = num_vertices;
	hg->num_edges = num_edges;

	hg->positions = (float*) calloc(num_vertices * 3, sizeof(float));
	hg->colors = (float*) calloc(num_vertices * 3, sizeof(float));
	hg->potentials = (float*) calloc(num_vertices, sizeof(float));
	hg->connections = (int*) calloc(num_edges * 2, sizeof(int));

	if (!hg->positions || !hg->colors || !hg->potentials || !hg->connections) {
		free(hg->positions);
		free(hg->colors);
		free(hg->potentials);
		free(hg->connections);
		free(hg);
		return NULL;
	}

	return hg;
}

/*=============================================================================
 * Thread Pool Implementation
 *===========================================================================*/

static void* thread_worker(void *arg) {
	ThreadPool *pool = (ThreadPool*) arg;

	while (pool->running) {
		/* Simple worker - could be expanded */
		sched_yield();
	}

	return NULL;
}

static ThreadPool* thread_pool_create(int num_threads) {
	ThreadPool *pool;
	int i;

	pool = (ThreadPool*) calloc(1, sizeof(ThreadPool));
	if (!pool)
		return NULL;

	pool->num_threads = num_threads;
	pool->running = 1;

	pool->threads = (pthread_t*) calloc(num_threads, sizeof(pthread_t));
	if (!pool->threads) {
		free(pool);
		return NULL;
	}

	for (i = 0; i < num_threads; i++) {
		pthread_create(&pool->threads[i], NULL, thread_worker, pool);
	}

	return pool;
}

static void thread_pool_destroy(ThreadPool *pool) {
	int i;

	if (!pool)
		return;

	pool->running = 0;

	for (i = 0; i < pool->num_threads; i++) {
		pthread_join(pool->threads[i], NULL);
	}

	free(pool->threads);
	free(pool);
}

/*=============================================================================
 * Game Initialization
 *===========================================================================*/

static NeuralGameState* game_create(void) {
	NeuralGameState *game;
	int layer_sizes[] = { INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS };
	int i;

	game = (NeuralGameState*) calloc(1, sizeof(NeuralGameState));
	if (!game)
		return NULL;

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

	/* Initialize invaders */
	game->num_invaders = MAX_INVADERS;
	for (i = 0; i < MAX_INVADERS; i++) {
		game->invader_x[i] = 100.0f + (i % 11) * 70.0f;
		game->invader_y[i] = 100.0f + (i / 11) * 60.0f;
		game->invader_health[i] = 100.0f;
		game->invader_type[i] = i % 3;
	}

	/* Initialize particles */
	for (i = 0; i < 1000 && i < MAX_PARTICLES; i++) {
		game->particles[i].x = random_float(0.0f, SCREEN_WIDTH);
		game->particles[i].y = random_float(0.0f, SCREEN_HEIGHT);
		game->particles[i].z = random_float(-100.0f, 100.0f);
		game->particles[i].vx = random_float(-10.0f, 10.0f);
		game->particles[i].vy = random_float(-10.0f, 10.0f);
		game->particles[i].vz = random_float(-5.0f, 5.0f);
		game->particles[i].life = 5.0f;
		game->particles[i].color = 0x80FFFF00; /* Yellow with alpha */
		game->particles[i].active = 1;
		game->num_particles++;
	}

	game->neural_field = 1.0f;
	game->wave = 1;
	game->delta_time = 1.0f / TARGET_FPS;

	return game;
}

/*=============================================================================
 * Game Update Functions
 *===========================================================================*/

static void game_update_ai(NeuralGameState *game) {
	double inputs[INPUT_NEURONS];
	double fuzzy_inputs[5];
	int i;

	if (!game || !game->neural_net || !game->fuzzy_sys)
		return;

	/* Prepare neural network inputs */
	for (i = 0; i < INPUT_NEURONS; i++) {
		inputs[i] = 0.0;
	}

	inputs[0] = game->player_x / SCREEN_WIDTH;
	inputs[1] = game->player_y / SCREEN_HEIGHT;
	inputs[2] = game->player_energy / 100.0;
	inputs[3] = game->player_shield / 100.0;

	for (i = 0; i < game->num_invaders && i < 20; i++) {
		if (game->invader_health[i] > 0) {
			inputs[4 + i * 2] = game->invader_x[i] / SCREEN_WIDTH;
			inputs[5 + i * 2] = game->invader_y[i] / SCREEN_HEIGHT;
		}
	}

	inputs[50] = game->neural_field;
	inputs[51] = game->entropy;

	/* Run neural network */
	neural_network_forward(game->neural_net, inputs);

	/* Prepare fuzzy inputs */
	fuzzy_inputs[0] = game->player_energy / 100.0;
	fuzzy_inputs[1] = game->player_shield / 100.0;
	fuzzy_inputs[2] = (double) game->num_invaders / MAX_INVADERS;
	fuzzy_inputs[3] = game->neural_field;
	fuzzy_inputs[4] = game->entropy;

	/* Run fuzzy inference */
	fuzzy_system_infer(game->fuzzy_sys, fuzzy_inputs);

	/* Update weapon cooldown */
	if (game->weapon_cooldown > 0.0f) {
		game->weapon_cooldown -= (float) game->delta_time;
	}
}

static void game_update_particles(NeuralGameState *game) {
	int i;

	for (i = 0; i < game->num_particles; i++) {
		SynapticParticle *p = &game->particles[i];

		if (!p->active)
			continue;

		/* Simple physics */
		p->vx += p->ax * (float) game->delta_time * 60.0f;
		p->vy += p->ay * (float) game->delta_time * 60.0f;
		p->vz += p->az * (float) game->delta_time * 60.0f;

		p->x += p->vx * (float) game->delta_time * 60.0f;
		p->y += p->vy * (float) game->delta_time * 60.0f;
		p->z += p->vz * (float) game->delta_time * 60.0f;

		p->life -= (float) game->delta_time;

		/* Boundary checking */
		if (p->x < 0 || p->x > SCREEN_WIDTH || p->y < 0 || p->y > SCREEN_HEIGHT
				|| p->life <= 0.0f) {
			p->active = 0;
		}
	}
}

/*=============================================================================
 * Rendering Functions
 *===========================================================================*/

static void draw_rect(float x, float y, float w, float h, unsigned int color) {
	float r = ((color >> 16) & 0xFF) / 255.0f;
	float g = ((color >> 8) & 0xFF) / 255.0f;
	float b = (color & 0xFF) / 255.0f;
	float a = ((color >> 24) & 0xFF) / 255.0f;

	glColor4f(r, g, b, a);
	glBegin(GL_QUADS);
	glVertex2f(x - w / 2, y - h / 2);
	glVertex2f(x + w / 2, y - h / 2);
	glVertex2f(x + w / 2, y + h / 2);
	glVertex2f(x - w / 2, y + h / 2);
	glEnd();
}

static void draw_circle(float x, float y, float r, unsigned int color) {
	int i;
	int segments = 20;
	float rf = ((color >> 16) & 0xFF) / 255.0f;
	float gf = ((color >> 8) & 0xFF) / 255.0f;
	float bf = (color & 0xFF) / 255.0f;
	float af = ((color >> 24) & 0xFF) / 255.0f;

	glColor4f(rf, gf, bf, af);
	glBegin(GL_TRIANGLE_FAN);
	glVertex2f(x, y);
	for (i = 0; i <= segments; i++) {
		float angle = i * 2.0f * (float) M_PI / segments;
		glVertex2f(x + cosf(angle) * r, y + sinf(angle) * r);
	}
	glEnd();
}

static void draw_particles(NeuralGameState *game) {
	int i;

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);
	glPointSize(2.0f);

	glBegin(GL_POINTS);
	for (i = 0; i < game->num_particles; i++) {
		if (game->particles[i].active) {
			unsigned int c = game->particles[i].color;
			float r = ((c >> 16) & 0xFF) / 255.0f;
			float g = ((c >> 8) & 0xFF) / 255.0f;
			float b = (c & 0xFF) / 255.0f;
			float a = ((c >> 24) & 0xFF) / 255.0f
					* (game->particles[i].life / 5.0f);

			glColor4f(r, g, b, a);
			glVertex2f(game->particles[i].x, game->particles[i].y);
		}
	}
	glEnd();

	glDisable(GL_BLEND);
}

static void draw_text(int x, int y, const char *text, unsigned int color,
		float scale) {
	/* Simple placeholder - in production use SDL_ttf */
	(void) x;
	(void) y;
	(void) text;
	(void) color;
	(void) scale;
}

static void render_game(NeuralGameState *game) {
	int i;
	char hud[256];

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	/* Set up 2D orthographic projection */
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0, SCREEN_WIDTH, SCREEN_HEIGHT, 0, -1, 1);
	glMatrixMode(GL_MODELVIEW);

	/* Draw background */
	glColor4f(0.02f, 0.02f, 0.05f, 1.0f);
	glBegin(GL_QUADS);
	glVertex2f(0, 0);
	glVertex2f(SCREEN_WIDTH, 0);
	glVertex2f(SCREEN_WIDTH, SCREEN_HEIGHT);
	glVertex2f(0, SCREEN_HEIGHT);
	glEnd();

	/* Draw particles */
	draw_particles(game);

	/* Draw invaders */
	for (i = 0; i < game->num_invaders; i++) {
		if (game->invader_health[i] > 0) {
			draw_circle(game->invader_x[i], game->invader_y[i], 15.0f,
					0xFF00FF00);
		}
	}

	/* Draw player */
	draw_rect(game->player_x, game->player_y, 30.0f, 30.0f, 0xFF0000FF);

	/* Draw shield if active */
	if (game->player_shield > 50.0f) {
		draw_circle(game->player_x, game->player_y, 40.0f, 0x80FFFF00);
	}

	/* Draw HUD */
	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

	sprintf(hud, "Score: %d  Wave: %d  Energy: %.0f  Shield: %.0f", game->score,
			game->wave, game->player_energy, game->player_shield);
	draw_text(10, 20, hud, 0xFFFFFFFF, 1.0f);

	sprintf(hud, "Neural Field: %.2f  Entropy: %.3f", game->neural_field,
			game->entropy);
	draw_text(10, 45, hud, 0xFFFFFFFF, 1.0f);

	/* Restore matrices */
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
}

/*=============================================================================
 * Game Loop
 *===========================================================================*/

static void game_loop(NeuralGameState *game) {
	Uint32 last_time, current_time;
	SDL_Event event;

	last_time = SDL_GetTicks();

	while (g_running && !game->game_over) {
		current_time = SDL_GetTicks();
		game->delta_time = (current_time - last_time) / 1000.0;
		last_time = current_time;

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
					if (game->weapon_cooldown <= 0.0f) {
						game->weapon_cooldown = 0.5f;
					}
					break;
				case SDLK_p:
					game->paused = !game->paused;
					break;
				}
			}
		}

		/* Handle player input */
		if (!game->paused) {
			const Uint8 *keys = SDL_GetKeyboardState(NULL);

			if (keys[SDL_SCANCODE_LEFT] || keys[SDL_SCANCODE_A]) {
				game->player_x -= 5.0f;
			}
			if (keys[SDL_SCANCODE_RIGHT] || keys[SDL_SCANCODE_D]) {
				game->player_x += 5.0f;
			}
			if (keys[SDL_SCANCODE_UP] || keys[SDL_SCANCODE_W]) {
				game->player_y -= 5.0f;
			}
			if (keys[SDL_SCANCODE_DOWN] || keys[SDL_SCANCODE_S]) {
				game->player_y += 5.0f;
			}

			/* Keep player on screen */
			if (game->player_x < 20)
				game->player_x = 20;
			if (game->player_x > SCREEN_WIDTH - 20)
				game->player_x = SCREEN_WIDTH - 20;
			if (game->player_y < 20)
				game->player_y = 20;
			if (game->player_y > SCREEN_HEIGHT - 20)
				game->player_y = SCREEN_HEIGHT - 20;

			/* Update game systems */
			game_update_ai(game);
			game_update_particles(game);
		}

		/* Render */
		render_game(game);
		SDL_GL_SwapWindow(g_window);

		/* Frame rate limiting */
		if (current_time - last_time < 16) {
			SDL_Delay(16 - (current_time - last_time));
		}

		game->frame_count++;
	}
}

/*=============================================================================
 * Game Cleanup
 *===========================================================================*/

static void game_destroy(NeuralGameState *game) {
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
		free(game->fuzzy_sys->inputs);
		free(game->fuzzy_sys->outputs);
		free(game->fuzzy_sys->rules);
		free(game->fuzzy_sys);
	}

	if (game->hyper_graph) {
		free(game->hyper_graph->positions);
		free(game->hyper_graph->colors);
		free(game->hyper_graph->potentials);
		free(game->hyper_graph->connections);
		free(game->hyper_graph);
	}

	free(game);
}

/*=============================================================================
 * Main Function
 *===========================================================================*/

int main(int argc, char *argv[]) {
	(void) argc;
	(void) argv;

	/* Initialize signal handlers */
	signal(SIGINT, signal_handler);
	signal(SIGTERM, signal_handler);

	printf("\n");
	printf("========================================\n");
	printf("  SpaceX - Advanced AI Core System\n");
	printf("  Version 4.0.1 - C90 Compliant\n");
	printf("========================================\n\n");

	/* Initialize SDL */
	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		fprintf(stderr, "SDL initialization failed: %s\n", SDL_GetError());
		return 1;
	}

	/* Create window */
	g_window = SDL_CreateWindow("SpaceX - Neural AI",
	SDL_WINDOWPOS_CENTERED,
	SDL_WINDOWPOS_CENTERED,
	SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);

	if (!g_window) {
		fprintf(stderr, "Window creation failed: %s\n", SDL_GetError());
		SDL_Quit();
		return 1;
	}

	/* Create OpenGL context */
	g_gl_context = SDL_GL_CreateContext(g_window);
	SDL_GL_SetSwapInterval(1);

	/* Initialize OpenGL */
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glClearColor(0.02f, 0.02f, 0.05f, 1.0f);

	/* Create thread pool */
	printf("Creating thread pool...\n");
	g_thread_pool = thread_pool_create(sysconf(_SC_NPROCESSORS_ONLN));
	if (g_thread_pool) {
		printf("  Thread pool created with %d threads\n",
				g_thread_pool->num_threads);
	}

	/* Create game */
	printf("Initializing game...\n");
	g_game = game_create();
	if (!g_game) {
		fprintf(stderr, "Game initialization failed\n");
		SDL_GL_DeleteContext(g_gl_context);
		SDL_DestroyWindow(g_window);
		SDL_Quit();
		return 1;
	}

	printf("\nGame initialized with %d particles\n", g_game->num_particles);
	printf("\nControls:\n");
	printf("  Arrow Keys / WASD - Move\n");
	printf("  SPACE - Fire\n");
	printf("  P - Pause\n");
	printf("  ESC - Exit\n\n");

	/* Run game */
	game_loop(g_game);

	/* Cleanup */
	printf("\nShutting down...\n");

	game_destroy(g_game);
	thread_pool_destroy(g_thread_pool);

	SDL_GL_DeleteContext(g_gl_context);
	SDL_DestroyWindow(g_window);
	SDL_Quit();

	printf("System terminated.\n");
	return 0;
}
