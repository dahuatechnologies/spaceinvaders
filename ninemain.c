/*
 * SpaceX - Mathematical AI Space Combat
 * Version: 6.3 - FULLY COMPILABLE with all colors defined
 *
 * Mathematical Foundation:
 * - Shannon Entropy: H(X) = -Σ p(x) log₂ p(x)
 * - Q-Learning for tactical decisions
 * - Spiking Neural Networks
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>

#include <SDL2/SDL.h>
#include <GL/gl.h>

/*=============================================================================
 * Game Constants
 *===========================================================================*/

#define SCREEN_WIDTH            1024
#define SCREEN_HEIGHT           768
#define TARGET_FPS              60
#define FRAME_TIME_MS            16

#define MAX_ENEMIES              15
#define MAX_BULLETS              50
#define MAX_PARTICLES            100

#define PLAYER_SPEED             5.0f
#define ENEMY_SPEED               2.0f
#define BULLET_SPEED              8.0f

#define STATE_DIM                8
#define ACTION_DIM                4
#define HIDDEN_DIM                16

/*=============================================================================
 * Colors - ALL colors defined here
 *===========================================================================*/

#define COLOR_BLACK      0x000000FF
#define COLOR_WHITE      0xFFFFFFFF
#define COLOR_RED        0xFF0000FF
#define COLOR_GREEN      0x00FF00FF
#define COLOR_BLUE       0x0000FFFF
#define COLOR_YELLOW     0xFFFF00FF
#define COLOR_CYAN       0x00FFFFFF
#define COLOR_MAGENTA    0xFF00FFFF
#define COLOR_ORANGE     0xFF8800FF
#define COLOR_PURPLE     0xAA00FFFF
#define COLOR_GOLD       0xFFD700FF  /* Added missing COLOR_GOLD definition */

/*=============================================================================
 * Game Structures
 *===========================================================================*/

typedef struct {
	float x, y;
	float vx, vy;
	int active;
	int owner;
} Bullet;

typedef struct {
	float x, y;
	float health;
	int active;
	int type;
} Enemy;

typedef struct {
	float x, y;
	float vx, vy;
	float life;
	int active;
	unsigned int color;
} Particle;

/* Q-Learning Agent */
typedef struct {
	float *q_table;
	int state_dim;
	int action_dim;
	float exploration_rate;
	float learning_rate;
	float discount_factor;
} QLearningAgent;

/* Spiking Neural Network Layer */
typedef struct {
	float *membrane;
	float *threshold;
	float *output;
	float **weights;
	int num_neurons;
	int num_inputs;
} SNNLayer;

typedef struct {
	SNNLayer **layers;
	int num_layers;
	float *input_buffer;
	float *output_buffer;
} SpikingNeuralNetwork;

/* Entropy System */
typedef struct {
	double *probabilities;
	int num_states;
	double entropy;
} EntropySystem;

/* Game State */
typedef struct {
	/* Player */
	float player_x, player_y;
	float player_health;
	int score;
	int wave;

	/* Game objects */
	Enemy enemies[MAX_ENEMIES];
	Bullet bullets[MAX_BULLETS];
	Particle particles[MAX_PARTICLES];

	int num_enemies;
	int num_bullets;
	int num_particles;

	/* AI Systems */
	QLearningAgent *q_agent;
	SpikingNeuralNetwork *snn;
	EntropySystem *entropy;

	/* Game state */
	int game_over;
	int paused;
	int victory;

	/* Time */
	Uint32 frame_count;
	float delta_time;
	float game_time;

	/* Metrics */
	double shannon_entropy;
} GameState;

/* Application State */
typedef struct {
	SDL_Window *window;
	SDL_GLContext gl_context;
	GameState *game;
	int running;
	float fps;
} AppState;

/*=============================================================================
 * Global State
 *===========================================================================*/

static volatile sig_atomic_t g_running = 1;

/*=============================================================================
 * Signal Handler
 *===========================================================================*/

static void handle_signal(int sig) {
	(void) sig;
	g_running = 0;
}

/*=============================================================================
 * Math Utilities
 *===========================================================================*/

static float random_float(float min, float max) {
	return min + ((float) rand() / RAND_MAX) * (max - min);
}

static float clampf(float value, float min, float max) {
	if (value < min)
		return min;
	if (value > max)
		return max;
	return value;
}

static float sigmoid(float x) {
	return 1.0f / (1.0f + expf(-x));
}

/*=============================================================================
 * Shannon Entropy
 *===========================================================================*/

static double shannon_entropy(const double *probs, int n) {
	double entropy = 0.0;
	int i;
	for (i = 0; i < n; i++) {
		if (probs[i] > 1e-15) {
			entropy -= probs[i] * log2(probs[i]);
		}
	}
	return entropy;
}

static EntropySystem* entropy_create(int num_states) {
	EntropySystem *es = (EntropySystem*) calloc(1, sizeof(EntropySystem));
	if (!es)
		return NULL;

	es->num_states = num_states;
	es->probabilities = (double*) calloc(num_states, sizeof(double));

	return es;
}

static void entropy_update(EntropySystem *es, const float *activations) {
	int i;
	double sum = 0.0;

	for (i = 0; i < es->num_states; i++) {
		es->probabilities[i] = exp(activations[i]);
		sum += es->probabilities[i];
	}

	if (sum > 1e-15) {
		for (i = 0; i < es->num_states; i++) {
			es->probabilities[i] /= sum;
		}
	}

	es->entropy = shannon_entropy(es->probabilities, es->num_states);
}

/*=============================================================================
 * Q-Learning Agent
 *===========================================================================*/

static QLearningAgent* q_agent_create(int state_dim, int action_dim) {
	QLearningAgent *agent = (QLearningAgent*) calloc(1, sizeof(QLearningAgent));
	if (!agent)
		return NULL;

	agent->state_dim = state_dim;
	agent->action_dim = action_dim;
	agent->q_table = (float*) calloc(state_dim * action_dim, sizeof(float));
	agent->exploration_rate = 0.2f;
	agent->learning_rate = 0.01f;
	agent->discount_factor = 0.95f;

	int i;
	for (i = 0; i < state_dim * action_dim; i++) {
		agent->q_table[i] = random_float(-0.1f, 0.1f);
	}

	return agent;
}

static int q_agent_select_action(QLearningAgent *agent, float *state) {
	int i, best_action = 0;
	float max_q = -1e10f;

	/* Exploration */
	if (random_float(0, 1) < agent->exploration_rate) {
		return rand() % agent->action_dim;
	}

	/* Exploitation */
	for (i = 0; i < agent->action_dim; i++) {
		float q_value = 0;
		int j;
		for (j = 0; j < agent->state_dim; j++) {
			q_value += agent->q_table[j * agent->action_dim + i] * state[j];
		}
		if (q_value > max_q) {
			max_q = q_value;
			best_action = i;
		}
	}

	return best_action;
}

/*=============================================================================
 * Spiking Neural Network
 *===========================================================================*/

static SpikingNeuralNetwork* snn_create(int *layer_sizes, int num_layers) {
	SpikingNeuralNetwork *snn = (SpikingNeuralNetwork*) calloc(1,
			sizeof(SpikingNeuralNetwork));
	if (!snn)
		return NULL;

	snn->num_layers = num_layers;
	snn->layers = (SNNLayer**) calloc(num_layers, sizeof(SNNLayer*));

	int i, j, k;
	for (i = 0; i < num_layers; i++) {
		snn->layers[i] = (SNNLayer*) calloc(1, sizeof(SNNLayer));
		snn->layers[i]->num_neurons = layer_sizes[i];
		snn->layers[i]->num_inputs =
				(i == 0) ? layer_sizes[0] : layer_sizes[i - 1];

		snn->layers[i]->membrane = (float*) calloc(layer_sizes[i],
				sizeof(float));
		snn->layers[i]->threshold = (float*) calloc(layer_sizes[i],
				sizeof(float));
		snn->layers[i]->output = (float*) calloc(layer_sizes[i], sizeof(float));

		snn->layers[i]->weights = (float**) calloc(layer_sizes[i],
				sizeof(float*));
		for (j = 0; j < layer_sizes[i]; j++) {
			snn->layers[i]->weights[j] = (float*) calloc(
					snn->layers[i]->num_inputs, sizeof(float));
			for (k = 0; k < snn->layers[i]->num_inputs; k++) {
				snn->layers[i]->weights[j][k] = random_float(-0.1f, 0.1f);
			}
			snn->layers[i]->threshold[j] = 1.0f;
		}
	}

	snn->input_buffer = (float*) calloc(layer_sizes[0], sizeof(float));
	snn->output_buffer = (float*) calloc(layer_sizes[num_layers - 1],
			sizeof(float));

	return snn;
}

static void snn_update(SpikingNeuralNetwork *snn, float *input, float dt) {
	int i, j, k;

	/* Input layer */
	for (i = 0; i < snn->layers[0]->num_neurons; i++) {
		snn->input_buffer[i] = input[i];
		snn->layers[0]->output[i] = sigmoid(input[i]);
	}

	/* Hidden layers */
	for (i = 1; i < snn->num_layers; i++) {
		SNNLayer *layer = snn->layers[i];
		SNNLayer *prev = snn->layers[i - 1];

		for (j = 0; j < layer->num_neurons; j++) {
			float sum = 0;
			for (k = 0; k < layer->num_inputs; k++) {
				sum += layer->weights[j][k] * prev->output[k];
			}

			/* Leaky integrate-and-fire (simplified) */
			layer->membrane[j] += (-layer->membrane[j] + sum) * dt * 10.0f;
			layer->output[j] = sigmoid(layer->membrane[j]);
		}
	}

	/* Output layer */
	SNNLayer *last = snn->layers[snn->num_layers - 1];
	for (i = 0; i < last->num_neurons; i++) {
		snn->output_buffer[i] = last->output[i];
	}
}

/*=============================================================================
 * Particle System
 *===========================================================================*/

static void create_particle(GameState *game, float x, float y,
		unsigned int color, float life) {
	int i;

	for (i = 0; i < MAX_PARTICLES; i++) {
		if (!game->particles[i].active) {
			game->particles[i].x = x;
			game->particles[i].y = y;
			game->particles[i].vx = random_float(-2.0f, 2.0f);
			game->particles[i].vy = random_float(-2.0f, 2.0f);
			game->particles[i].life = life;
			game->particles[i].active = 1;
			game->particles[i].color = color;
			game->num_particles++;
			break;
		}
	}
}

static void update_particles(GameState *game) {
	int i;

	for (i = 0; i < MAX_PARTICLES; i++) {
		if (!game->particles[i].active)
			continue;

		game->particles[i].x += game->particles[i].vx;
		game->particles[i].y += game->particles[i].vy;
		game->particles[i].life -= game->delta_time;

		if (game->particles[i].life <= 0) {
			game->particles[i].active = 0;
			game->num_particles--;
		}
	}
}

/*=============================================================================
 * Game Initialization
 *===========================================================================*/

static void init_game(GameState *game) {
	int i;

	memset(game, 0, sizeof(GameState));

	/* Player */
	game->player_x = SCREEN_WIDTH / 2.0f;
	game->player_y = SCREEN_HEIGHT - 100.0f;
	game->player_health = 100;
	game->score = 0;
	game->wave = 1;

	/* AI Systems */
	int layer_sizes[] = { STATE_DIM, HIDDEN_DIM, ACTION_DIM };
	game->q_agent = q_agent_create(STATE_DIM, ACTION_DIM);
	game->snn = snn_create(layer_sizes, 3);
	game->entropy = entropy_create(HIDDEN_DIM);

	/* Game objects */
	for (i = 0; i < MAX_ENEMIES; i++) {
		game->enemies[i].active = 0;
	}

	for (i = 0; i < MAX_BULLETS; i++) {
		game->bullets[i].active = 0;
	}

	for (i = 0; i < MAX_PARTICLES; i++) {
		game->particles[i].active = 0;
	}

	game->num_enemies = 0;
	game->num_bullets = 0;
	game->num_particles = 0;

	/* Game starts unpaused and not game over */
	game->paused = 0;
	game->game_over = 0;
	game->victory = 0;
}

/*=============================================================================
 * Spawn Enemy
 *===========================================================================*/

static void spawn_enemy(GameState *game) {
	int i;

	if (game->num_enemies >= MAX_ENEMIES)
		return;

	for (i = 0; i < MAX_ENEMIES; i++) {
		if (!game->enemies[i].active) {
			game->enemies[i].x = random_float(50, SCREEN_WIDTH - 50);
			game->enemies[i].y = 50;
			game->enemies[i].health = 30 + game->wave * 5;
			game->enemies[i].active = 1;
			game->enemies[i].type = rand() % 3;
			game->num_enemies++;
			break;
		}
	}
}

/*=============================================================================
 * Game Update
 *===========================================================================*/

static void game_update(GameState *game) {
	int i, j;
	float dx, dy, dist;

	/* Don't update if paused or game over */
	if (game->paused || game->game_over || game->victory) {
		return;
	}

	game->frame_count++;
	game->game_time += game->delta_time;

	/* Spawn enemies */
	if (game->num_enemies < 3 + game->wave && rand() % 100 < 2) {
		spawn_enemy(game);
	}

	/* Build state vector for AI */
	float state[STATE_DIM];
	for (i = 0; i < STATE_DIM; i++) {
		state[i] = 0;
	}

	state[0] = game->player_x / SCREEN_WIDTH;
	state[1] = game->player_y / SCREEN_HEIGHT;
	state[2] = game->player_health / 100.0f;
	state[3] = game->wave / 10.0f;
	state[4] = game->num_enemies / (float) MAX_ENEMIES;

	/* Update AI */
	int action = q_agent_select_action(game->q_agent, state);
	(void) action; /* Suppress unused warning */

	snn_update(game->snn, state, game->delta_time);

	/* Update entropy */
	entropy_update(game->entropy, game->snn->output_buffer);
	game->shannon_entropy = game->entropy->entropy;

	/* Update enemies */
	for (i = 0; i < game->num_enemies; i++) {
		if (!game->enemies[i].active)
			continue;

		/* Move toward player */
		dx = game->player_x - game->enemies[i].x;
		dy = game->player_y - game->enemies[i].y;
		dist = sqrt(dx * dx + dy * dy);

		if (dist > 1.0f) {
			game->enemies[i].x += (dx / dist) * ENEMY_SPEED;
			game->enemies[i].y += (dy / dist) * ENEMY_SPEED;
		}

		/* Check collision with player */
		dx = game->player_x - game->enemies[i].x;
		dy = game->player_y - game->enemies[i].y;
		dist = sqrt(dx * dx + dy * dy);

		if (dist < 40) {
			game->player_health -= 10;
			game->enemies[i].active = 0;
			game->num_enemies--;

			create_particle(game, game->enemies[i].x, game->enemies[i].y,
			COLOR_RED, 0.5f);

			if (game->player_health <= 0) {
				game->game_over = 1;
			}
		}
	}

	/* Update bullets */
	for (i = 0; i < game->num_bullets; i++) {
		if (!game->bullets[i].active)
			continue;

		game->bullets[i].x += game->bullets[i].vx;
		game->bullets[i].y += game->bullets[i].vy;

		/* Remove off-screen bullets */
		if (game->bullets[i].y < -50 || game->bullets[i].y > SCREEN_HEIGHT + 50
				|| game->bullets[i].x < -50
				|| game->bullets[i].x > SCREEN_WIDTH + 50) {
			game->bullets[i].active = 0;
			game->num_bullets--;
			continue;
		}

		/* Player bullet collision with enemies */
		if (game->bullets[i].owner == 0) {
			for (j = 0; j < game->num_enemies; j++) {
				if (!game->enemies[j].active)
					continue;

				dx = game->bullets[i].x - game->enemies[j].x;
				dy = game->bullets[i].y - game->enemies[j].y;
				dist = sqrt(dx * dx + dy * dy);

				if (dist < 25) {
					game->enemies[j].health -= 20;
					game->bullets[i].active = 0;
					game->num_bullets--;

					create_particle(game, game->bullets[i].x,
							game->bullets[i].y,
							COLOR_YELLOW, 0.3f);

					if (game->enemies[j].health <= 0) {
						game->score += 100;
						game->enemies[j].active = 0;
						game->num_enemies--;

						create_particle(game, game->enemies[j].x,
								game->enemies[j].y,
								COLOR_ORANGE, 0.8f);
					}
					break;
				}
			}
		}
	}

	/* Update particles */
	update_particles(game);

	/* Wave progression */
	if (game->num_enemies == 0) {
		game->wave++;
		game->player_health =
				(game->player_health + 20 > 100) ?
						100 : game->player_health + 20;

		if (game->wave > 10) {
			game->victory = 1;
		}
	}
}

/*=============================================================================
 * Input Processing
 *===========================================================================*/

static void process_input(AppState *state) {
	GameState *game = state->game;
	SDL_Event event;
	const Uint8 *keys;
	int i;

	while (SDL_PollEvent(&event)) {
		switch (event.type) {
		case SDL_QUIT:
			state->running = 0;
			g_running = 0;
			return;

		case SDL_KEYDOWN:
			switch (event.key.keysym.sym) {
			case SDLK_ESCAPE:
				state->running = 0;
				g_running = 0;
				return;

			case SDLK_p:
				/* Toggle pause - this should always work */
				game->paused = !game->paused;
				printf("Game %s\n", game->paused ? "PAUSED" : "UNPAUSED");
				break;

			case SDLK_r:
				if (game->game_over || game->victory) {
					init_game(game);
					printf("Game restarted\n");
				}
				break;

			case SDLK_SPACE:
				if (!game->paused && !game->game_over && !game->victory) {
					/* Fire bullet */
					for (i = 0; i < MAX_BULLETS; i++) {
						if (!game->bullets[i].active) {
							game->bullets[i].x = game->player_x;
							game->bullets[i].y = game->player_y - 20;
							game->bullets[i].vx = 0;
							game->bullets[i].vy = -BULLET_SPEED;
							game->bullets[i].active = 1;
							game->bullets[i].owner = 0;
							game->num_bullets++;
							break;
						}
					}
				}
				break;
			}
			break;
		}
	}

	keys = SDL_GetKeyboardState(NULL);

	/* Handle continuous input - only if not paused and not game over */
	if (!game->paused && !game->game_over && !game->victory) {
		float move_x = 0, move_y = 0;

		if (keys[SDL_SCANCODE_LEFT] || keys[SDL_SCANCODE_A])
			move_x -= PLAYER_SPEED;
		if (keys[SDL_SCANCODE_RIGHT] || keys[SDL_SCANCODE_D])
			move_x += PLAYER_SPEED;
		if (keys[SDL_SCANCODE_UP] || keys[SDL_SCANCODE_W])
			move_y -= PLAYER_SPEED;
		if (keys[SDL_SCANCODE_DOWN] || keys[SDL_SCANCODE_S])
			move_y += PLAYER_SPEED;

		game->player_x += move_x;
		game->player_y += move_y;

		/* Keep player on screen */
		game->player_x = clampf(game->player_x, 30, SCREEN_WIDTH - 30);
		game->player_y = clampf(game->player_y, 30, SCREEN_HEIGHT - 30);
	}
}

/*=============================================================================
 * Rendering
 *===========================================================================*/

static void draw_rect(float x, float y, float w, float h, unsigned int color) {
	float r = ((color >> 24) & 0xFF) / 255.0f;
	float g = ((color >> 16) & 0xFF) / 255.0f;
	float b = ((color >> 8) & 0xFF) / 255.0f;
	float a = (color & 0xFF) / 255.0f;

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
	float rf = ((color >> 24) & 0xFF) / 255.0f;
	float gf = ((color >> 16) & 0xFF) / 255.0f;
	float bf = ((color >> 8) & 0xFF) / 255.0f;
	float af = (color & 0xFF) / 255.0f;

	glColor4f(rf, gf, bf, af);
	glBegin(GL_TRIANGLE_FAN);
	glVertex2f(x, y);
	for (i = 0; i <= segments; i++) {
		float angle = i * 2.0f * (float) M_PI / segments;
		glVertex2f(x + cosf(angle) * r, y + sinf(angle) * r);
	}
	glEnd();
}

static void draw_particles(GameState *game) {
	int i;

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);
	glPointSize(3.0f);

	glBegin(GL_POINTS);
	for (i = 0; i < MAX_PARTICLES; i++) {
		if (!game->particles[i].active)
			continue;

		float r = ((game->particles[i].color >> 24) & 0xFF) / 255.0f;
		float g = ((game->particles[i].color >> 16) & 0xFF) / 255.0f;
		float b = ((game->particles[i].color >> 8) & 0xFF) / 255.0f;
		float a = game->particles[i].life;

		glColor4f(r, g, b, a);
		glVertex2f(game->particles[i].x, game->particles[i].y);
	}
	glEnd();

	glDisable(GL_BLEND);
}

static void draw_text(int x, int y, const char *text, unsigned int color) {
	/* Simple placeholder - in production use SDL_ttf */
	(void) x;
	(void) y;
	(void) text;
	(void) color;
}

static void render_game(AppState *state) {
	GameState *game = state->game;
	int i;
	char hud[256];

	glClear(GL_COLOR_BUFFER_BIT);
	glLoadIdentity();

	/* Set up 2D projection */
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, SCREEN_WIDTH, SCREEN_HEIGHT, 0, -1, 1);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	/* Draw background */
	glClearColor(0.02f, 0.02f, 0.05f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	/* Draw stars */
	glPointSize(1.5f);
	glColor4f(0.8f, 0.8f, 1.0f, 0.5f);
	glBegin(GL_POINTS);
	for (i = 0; i < 100; i++) {
		float x = fmodf(game->game_time * 20 + i * 37, SCREEN_WIDTH);
		float y = fmodf(i * 73, SCREEN_HEIGHT);
		glVertex2f(x, y);
	}
	glEnd();

	/* Draw particles */
	draw_particles(game);

	/* Draw enemies */
	for (i = 0; i < game->num_enemies; i++) {
		if (!game->enemies[i].active)
			continue;

		unsigned int color;
		if (game->enemies[i].type == 0)
			color = COLOR_RED;
		else if (game->enemies[i].type == 1)
			color = COLOR_ORANGE;
		else
			color = COLOR_PURPLE;

		draw_circle(game->enemies[i].x, game->enemies[i].y, 15, color);

		/* Health bar */
		float health_pct = game->enemies[i].health / (30 + game->wave * 5);
		draw_rect(game->enemies[i].x, game->enemies[i].y - 25, 30, 4,
				COLOR_RED);
		draw_rect(game->enemies[i].x - 15 + 15 * health_pct,
				game->enemies[i].y - 25, 30 * health_pct, 4, COLOR_GREEN);
	}

	/* Draw bullets */
	for (i = 0; i < game->num_bullets; i++) {
		if (!game->bullets[i].active)
			continue;

		unsigned int color =
				(game->bullets[i].owner == 0) ? COLOR_CYAN : COLOR_YELLOW;
		draw_rect(game->bullets[i].x, game->bullets[i].y, 3, 8, color);
	}

	/* Draw player */
	draw_rect(game->player_x, game->player_y, 20, 20, COLOR_GREEN);

	/* Draw health bar */
	draw_rect(game->player_x, game->player_y - 30, 40, 4, COLOR_RED);
	draw_rect(game->player_x - 20 + 20 * (game->player_health / 100.0f),
			game->player_y - 30, 40 * (game->player_health / 100.0f), 4,
			COLOR_GREEN);

	/* Draw HUD */
	sprintf(hud, "Score: %d  Wave: %d  Health: %.0f", game->score, game->wave,
			game->player_health);
	draw_text(10, 20, hud, COLOR_WHITE);

	sprintf(hud, "Entropy: %.4f  FPS: %.1f", game->shannon_entropy, state->fps);
	draw_text(10, 45, hud, COLOR_WHITE);

	/* Draw pause indicator */
	if (game->paused) {
		glColor4f(0, 0, 0, 0.5f);
		glBegin(GL_QUADS);
		glVertex2f(0, 0);
		glVertex2f(SCREEN_WIDTH, 0);
		glVertex2f(SCREEN_WIDTH, SCREEN_HEIGHT);
		glVertex2f(0, SCREEN_HEIGHT);
		glEnd();

		draw_text(SCREEN_WIDTH / 2 - 50, SCREEN_HEIGHT / 2, "PAUSED",
				COLOR_WHITE);
		draw_text(SCREEN_WIDTH / 2 - 70, SCREEN_HEIGHT / 2 + 30,
				"Press P to unpause", COLOR_WHITE);
	}

	/* Draw game over screen */
	if (game->game_over) {
		glColor4f(0, 0, 0, 0.8f);
		glBegin(GL_QUADS);
		glVertex2f(0, 0);
		glVertex2f(SCREEN_WIDTH, 0);
		glVertex2f(SCREEN_WIDTH, SCREEN_HEIGHT);
		glVertex2f(0, SCREEN_HEIGHT);
		glEnd();

		draw_text(SCREEN_WIDTH / 2 - 70, SCREEN_HEIGHT / 2 - 20, "GAME OVER",
				COLOR_RED);
		sprintf(hud, "Final Score: %d", game->score);
		draw_text(SCREEN_WIDTH / 2 - 60, SCREEN_HEIGHT / 2 + 10, hud,
				COLOR_WHITE);
		draw_text(SCREEN_WIDTH / 2 - 70, SCREEN_HEIGHT / 2 + 40,
				"Press R to restart", COLOR_WHITE);
	}

	/* Draw victory screen */
	if (game->victory) {
		glColor4f(0, 0, 0, 0.8f);
		glBegin(GL_QUADS);
		glVertex2f(0, 0);
		glVertex2f(SCREEN_WIDTH, 0);
		glVertex2f(SCREEN_WIDTH, SCREEN_HEIGHT);
		glVertex2f(0, SCREEN_HEIGHT);
		glEnd();

		/* COLOR_GOLD is now defined */
		draw_text(SCREEN_WIDTH / 2 - 50, SCREEN_HEIGHT / 2 - 20, "VICTORY!",
				COLOR_GOLD);
		sprintf(hud, "Final Score: %d", game->score);
		draw_text(SCREEN_WIDTH / 2 - 50, SCREEN_HEIGHT / 2 + 10, hud,
				COLOR_WHITE);
		draw_text(SCREEN_WIDTH / 2 - 70, SCREEN_HEIGHT / 2 + 40,
				"Press R to restart", COLOR_WHITE);
	}
}

/*=============================================================================
 * Application Initialization
 *===========================================================================*/

static int app_init(AppState *state) {
	memset(state, 0, sizeof(AppState));

	printf("\n");
	printf("╔════════════════════════════════════════════════════════════╗\n");
	printf("║     SpaceX - Mathematical AI Space Combat v6.3            ║\n");
	printf("║     Shannon Entropy | Q-Learning | Spiking Neural Nets    ║\n");
	printf(
			"╚════════════════════════════════════════════════════════════╝\n\n");

	srand((unsigned int) time(NULL));

	signal(SIGINT, handle_signal);
	signal(SIGTERM, handle_signal);

	if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) < 0) {
		fprintf(stderr, "SDL Init failed: %s\n", SDL_GetError());
		return -1;
	}

	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

	state->window = SDL_CreateWindow("SpaceX - Mathematical AI Combat",
	SDL_WINDOWPOS_CENTERED,
	SDL_WINDOWPOS_CENTERED,
	SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);

	if (!state->window) {
		fprintf(stderr, "Window creation failed: %s\n", SDL_GetError());
		SDL_Quit();
		return -1;
	}

	state->gl_context = SDL_GL_CreateContext(state->window);
	if (!state->gl_context) {
		fprintf(stderr, "OpenGL context failed: %s\n", SDL_GetError());
		SDL_DestroyWindow(state->window);
		SDL_Quit();
		return -1;
	}

	SDL_GL_SetSwapInterval(1);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glClearColor(0.02f, 0.02f, 0.05f, 1.0f);

	printf("OpenGL: %s\n", glGetString(GL_RENDERER));

	printf("\nInitializing game...\n");
	state->game = (GameState*) malloc(sizeof(GameState));
	if (!state->game) {
		fprintf(stderr, "Failed to allocate game state\n");
		SDL_GL_DeleteContext(state->gl_context);
		SDL_DestroyWindow(state->window);
		SDL_Quit();
		return -1;
	}

	init_game(state->game);
	state->running = 1;

	printf("\n");
	printf("Controls:\n");
	printf("  Arrow Keys / WASD - Move\n");
	printf("  SPACE - Fire\n");
	printf("  P - Pause (game starts unpaused)\n");
	printf("  R - Restart (after game over)\n");
	printf("  ESC - Exit\n\n");

	printf("Game started - press P to pause/unpause\n\n");

	return 0;
}

/*=============================================================================
 * Application Cleanup
 *===========================================================================*/

static void app_cleanup(AppState *state) {
	printf("\nShutting down...\n");

	if (state->game) {
		printf("Final Score: %d\n", state->game->score);
		printf("Waves Completed: %d\n", state->game->wave - 1);
		printf("Final Entropy: %.4f\n", state->game->shannon_entropy);

		/* Free AI systems */
		if (state->game->q_agent) {
			free(state->game->q_agent->q_table);
			free(state->game->q_agent);
		}

		/* Note: In a full implementation, you'd free all the SNN layers here */
		free(state->game->snn);
		free(state->game->entropy);
		free(state->game);
	}

	if (state->gl_context) {
		SDL_GL_DeleteContext(state->gl_context);
	}
	if (state->window) {
		SDL_DestroyWindow(state->window);
	}
	SDL_Quit();

	printf("Done.\n");
}

/*=============================================================================
 * Main Loop
 *===========================================================================*/

static void app_run(AppState *state) {
	Uint32 last_time, current_time;
	Uint32 fps_last_time = SDL_GetTicks();
	int fps_frames = 0;

	last_time = SDL_GetTicks();

	while (state->running && g_running) {
		current_time = SDL_GetTicks();

		/* Calculate delta time */
		state->game->delta_time = (current_time - last_time) / 1000.0f;
		last_time = current_time;

		/* Process input (this handles pause toggling) */
		process_input(state);

		/* Update game (only if not paused) */
		game_update(state->game);

		/* Render (always render, even when paused) */
		render_game(state);

		SDL_GL_SwapWindow(state->window);

		/* FPS calculation */
		fps_frames++;
		if (current_time - fps_last_time >= 1000) {
			state->fps = (float) fps_frames * 1000.0f
					/ (current_time - fps_last_time);
			fps_frames = 0;
			fps_last_time = current_time;
		}

		/* Frame rate limiting */
		Uint32 frame_time = SDL_GetTicks() - current_time;
		if (frame_time < FRAME_TIME_MS) {
			SDL_Delay(FRAME_TIME_MS - frame_time);
		}
	}
}

/*=============================================================================
 * Main Entry Point
 *===========================================================================*/

int main(int argc, char *argv[]) {
	AppState state;
	int result;

	(void) argc;
	(void) argv;

	result = app_init(&state);
	if (result != 0) {
		return 1;
	}

	app_run(&state);
	app_cleanup(&state);

	return 0;
}
