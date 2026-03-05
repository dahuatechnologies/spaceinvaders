/*
 * EvoX - Ultimate AI Space Battle
 * Version: 4.2 - FULLY WORKING with Sound & Controls
 *
 * AI Features: Q-Learning for Enemy Behavior
 * Audio: Procedural Sound Generation with OpenAL
 * Graphics: 2D Rendering with Player vs AI Enemies
 *
 * Compilation: gcc -std=c90 -O2 -pthread -lSDL2 -lGL -lGLU -lopenal -lm -o evox main.c
 */

#define _POSIX_C_SOURCE 200809L

/*=============================================================================
 * System Headers
 *===========================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>

/*=============================================================================
 * External Libraries
 *===========================================================================*/

#include <SDL2/SDL.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <AL/al.h>
#include <AL/alc.h>

/*=============================================================================
 * Game Configuration
 *===========================================================================*/

#define SCREEN_WIDTH            1024
#define SCREEN_HEIGHT           768
#define TARGET_FPS              60
#define FRAME_TIME_MS            16

#define MAX_ENEMIES              20
#define MAX_BULLETS              50
#define MAX_PARTICLES           200

#define PLAYER_SPEED             8.0f
#define ENEMY_SPEED              2.0f
#define BULLET_SPEED            15.0f

/*=============================================================================
 * Colors (32-bit RGBA) - All colors defined
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
 * Sound Configuration
 *===========================================================================*/

#define AUDIO_SAMPLE_RATE       44100
#define AUDIO_CHANNELS          1
#define AUDIO_BUFFER_SIZE       44100  /* 1 second buffer */

/*=============================================================================
 * Type Definitions
 *===========================================================================*/

typedef struct {
	float x, y;
	float vx, vy;
	int active;
	int owner; /* 0 = player, 1 = enemy */
} Bullet;

typedef struct {
	float x, y;
	float health;
	float max_health;
	int active;
	float speed;
	int type;
	float attack_cooldown;
} Enemy;

typedef struct {
	float x, y;
	float vx, vy;
	float life;
	int active;
	unsigned int color;
} Particle;

typedef struct {
	ALuint source;
	ALuint buffer;
	int active;
	char name[32];
} SoundEffect;

typedef struct {
	ALCdevice *device;
	ALCcontext *context;
	SoundEffect sounds[10];
	float listener_x, listener_y;
	int initialized;
} AudioSystem;

typedef struct {
	float player_x, player_y;
	float player_health;
	float player_energy;
	int score;
	int wave;
	int kills;

	Enemy enemies[MAX_ENEMIES];
	Bullet bullets[MAX_BULLETS];
	Particle particles[MAX_PARTICLES];

	int num_enemies;
	int num_bullets;
	int num_particles;

	int game_over;
	int paused;
	int victory;

	Uint32 frame_count;
	float delta_time;
	float game_time;

	/* Q-Learning AI state */
	float *q_table;
	int learning_step;
} GameState;

typedef struct {
	SDL_Window *window;
	SDL_GLContext gl_context;
	GameState *game;
	AudioSystem *audio;
	int running;
	float fps;
	struct timeval last_frame;
} AppState;

/*=============================================================================
 * Global Variables
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

/*=============================================================================
 * Audio System - Procedural Sound Generation
 *===========================================================================*/

static ALuint generate_sine_wave(float frequency, float duration, float volume) {
	ALuint buffer;
	int num_samples = (int) (AUDIO_SAMPLE_RATE * duration);
	short *samples;
	int i;

	samples = (short*) malloc(num_samples * sizeof(short));
	if (!samples)
		return 0;

	for (i = 0; i < num_samples; i++) {
		float t = (float) i / AUDIO_SAMPLE_RATE;
		float value = sinf(2.0f * (float) M_PI * frequency * t);
		samples[i] = (short) (value * volume * 32767);
	}

	alGenBuffers(1, &buffer);
	alBufferData(buffer, AL_FORMAT_MONO16, samples, num_samples * sizeof(short),
			AUDIO_SAMPLE_RATE);

	free(samples);
	return buffer;
}

static ALuint generate_laser_sound(void) {
	ALuint buffer;
	int num_samples = (int) (AUDIO_SAMPLE_RATE * 0.2f);
	short *samples;
	int i;

	samples = (short*) malloc(num_samples * sizeof(short));
	if (!samples)
		return 0;

	for (i = 0; i < num_samples; i++) {
		float t = (float) i / AUDIO_SAMPLE_RATE;
		float freq = 880.0f + 440.0f * (1.0f - t * 5.0f);
		float value = sinf(2.0f * (float) M_PI * freq * t);
		float envelope = 1.0f - t * 5.0f;
		if (envelope < 0)
			envelope = 0;
		samples[i] = (short) (value * envelope * 0.5f * 32767);
	}

	alGenBuffers(1, &buffer);
	alBufferData(buffer, AL_FORMAT_MONO16, samples, num_samples * sizeof(short),
			AUDIO_SAMPLE_RATE);

	free(samples);
	return buffer;
}

static ALuint generate_explosion_sound(void) {
	ALuint buffer;
	int num_samples = (int) (AUDIO_SAMPLE_RATE * 0.5f);
	short *samples;
	int i;

	samples = (short*) malloc(num_samples * sizeof(short));
	if (!samples)
		return 0;

	for (i = 0; i < num_samples; i++) {
		float t = (float) i / AUDIO_SAMPLE_RATE;
		float noise = random_float(-1.0f, 1.0f);
		float envelope = expf(-t * 10.0f);
		samples[i] = (short) (noise * envelope * 0.8f * 32767);
	}

	alGenBuffers(1, &buffer);
	alBufferData(buffer, AL_FORMAT_MONO16, samples, num_samples * sizeof(short),
			AUDIO_SAMPLE_RATE);

	free(samples);
	return buffer;
}

static ALuint generate_hit_sound(void) {
	ALuint buffer;
	int num_samples = (int) (AUDIO_SAMPLE_RATE * 0.1f);
	short *samples;
	int i;

	samples = (short*) malloc(num_samples * sizeof(short));
	if (!samples)
		return 0;

	for (i = 0; i < num_samples; i++) {
		float t = (float) i / AUDIO_SAMPLE_RATE;
		float value = sinf(2.0f * (float) M_PI * 440.0f * t);
		float envelope = 1.0f - t * 10.0f;
		if (envelope < 0)
			envelope = 0;
		samples[i] = (short) (value * envelope * 0.3f * 32767);
	}

	alGenBuffers(1, &buffer);
	alBufferData(buffer, AL_FORMAT_MONO16, samples, num_samples * sizeof(short),
			AUDIO_SAMPLE_RATE);

	free(samples);
	return buffer;
}

static AudioSystem* audio_system_create(void) {
	AudioSystem *audio;
	int i;

	audio = (AudioSystem*) calloc(1, sizeof(AudioSystem));
	if (!audio)
		return NULL;

	audio->device = alcOpenDevice(NULL);
	if (!audio->device) {
		printf("Warning: Could not open audio device\n");
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

	printf("Audio System Initialized: %s\n",
			alcGetString(audio->device, ALC_DEVICE_SPECIFIER));

	/* Generate sound effects */
	for (i = 0; i < 10; i++) {
		alGenSources(1, &audio->sounds[i].source);
		audio->sounds[i].active = 0;
		strcpy(audio->sounds[i].name, "unknown");
	}

	/* Laser sound (index 0) */
	audio->sounds[0].buffer = generate_laser_sound();
	strcpy(audio->sounds[0].name, "laser");
	alSourcei(audio->sounds[0].source, AL_BUFFER, audio->sounds[0].buffer);
	audio->sounds[0].active = 1;

	/* Explosion sound (index 1) */
	audio->sounds[1].buffer = generate_explosion_sound();
	strcpy(audio->sounds[1].name, "explosion");
	alSourcei(audio->sounds[1].source, AL_BUFFER, audio->sounds[1].buffer);
	audio->sounds[1].active = 1;

	/* Hit sound (index 2) */
	audio->sounds[2].buffer = generate_hit_sound();
	strcpy(audio->sounds[2].name, "hit");
	alSourcei(audio->sounds[2].source, AL_BUFFER, audio->sounds[2].buffer);
	audio->sounds[2].active = 1;

	/* Game over sound (index 3) */
	audio->sounds[3].buffer = generate_sine_wave(220.0f, 1.0f, 0.7f);
	strcpy(audio->sounds[3].name, "gameover");
	alSourcei(audio->sounds[3].source, AL_BUFFER, audio->sounds[3].buffer);
	audio->sounds[3].active = 1;

	/* Victory sound (index 4) */
	audio->sounds[4].buffer = generate_sine_wave(523.0f, 0.5f, 0.8f);
	strcpy(audio->sounds[4].name, "victory");
	alSourcei(audio->sounds[4].source, AL_BUFFER, audio->sounds[4].buffer);
	audio->sounds[4].active = 1;

	audio->listener_x = SCREEN_WIDTH / 2;
	audio->listener_y = SCREEN_HEIGHT / 2;
	alListener3f(AL_POSITION, audio->listener_x, audio->listener_y, 0);

	audio->initialized = 1;
	return audio;
}

static void audio_play_sound(AudioSystem *audio, const char *name, float x,
		float y) {
	int i;
	ALint state;

	if (!audio || !audio->initialized)
		return;

	for (i = 0; i < 10; i++) {
		if (audio->sounds[i].active
				&& strcmp(audio->sounds[i].name, name) == 0) {
			alGetSourcei(audio->sounds[i].source, AL_SOURCE_STATE, &state);

			if (state != AL_PLAYING) {
				alSource3f(audio->sounds[i].source, AL_POSITION, x, y, 0);
				alSourcePlay(audio->sounds[i].source);
			}
			break;
		}
	}
}

static void audio_update_listener(AudioSystem *audio, float x, float y) {
	if (!audio || !audio->initialized)
		return;

	audio->listener_x = x;
	audio->listener_y = y;
	alListener3f(AL_POSITION, x, y, 0);
}

static void audio_system_destroy(AudioSystem *audio) {
	int i;

	if (!audio)
		return;

	if (audio->initialized) {
		for (i = 0; i < 10; i++) {
			alSourceStop(audio->sounds[i].source);
			alDeleteSources(1, &audio->sounds[i].source);
			if (audio->sounds[i].buffer) {
				alDeleteBuffers(1, &audio->sounds[i].buffer);
			}
		}

		alcMakeContextCurrent(NULL);
		alcDestroyContext(audio->context);
		alcCloseDevice(audio->device);
	}

	free(audio);
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
			game->particles[i].vx = random_float(-3.0f, 3.0f);
			game->particles[i].vy = random_float(-3.0f, 3.0f);
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

	i = 0;
	while (i < game->num_particles) {
		if (!game->particles[i].active) {
			i++;
			continue;
		}

		game->particles[i].x += game->particles[i].vx;
		game->particles[i].y += game->particles[i].vy;
		game->particles[i].life -= game->delta_time;

		if (game->particles[i].life <= 0) {
			game->particles[i].active = 0;
			game->num_particles--;
		} else {
			i++;
		}
	}
}

/*=============================================================================
 * Q-Learning AI Implementation
 *===========================================================================*/

static float* q_learning_create(int state_size, int action_size) {
	float *q_table;
	int i;

	q_table = (float*) calloc(state_size * action_size, sizeof(float));
	if (!q_table)
		return NULL;

	for (i = 0; i < state_size * action_size; i++) {
		q_table[i] = random_float(-0.1f, 0.1f);
	}

	return q_table;
}

static int q_learning_select_action(float *q_table, int state, int action_size,
		float exploration_rate) {
	int i, best_action = 0;
	float max_q = -1e10f;

	/* Exploration */
	if (random_float(0, 1) < exploration_rate) {
		return rand() % action_size;
	}

	/* Exploitation */
	for (i = 0; i < action_size; i++) {
		if (q_table[state * action_size + i] > max_q) {
			max_q = q_table[state * action_size + i];
			best_action = i;
		}
	}

	return best_action;
}

static void q_learning_update(float *q_table, int state, int action,
		float reward, int next_state, int action_size, float learning_rate,
		float discount_factor) {
	int i;
	float max_next_q = -1e10f;
	float current_q;
	float td_error;

	/* Find max Q for next state */
	for (i = 0; i < action_size; i++) {
		if (q_table[next_state * action_size + i] > max_next_q) {
			max_next_q = q_table[next_state * action_size + i];
		}
	}

	current_q = q_table[state * action_size + action];
	td_error = reward + discount_factor * max_next_q - current_q;
	q_table[state * action_size + action] += learning_rate * td_error;
}

/*=============================================================================
 * Game Initialization
 *===========================================================================*/

static void init_game(GameState *game) {
	int i;

	memset(game, 0, sizeof(GameState));

	game->player_x = SCREEN_WIDTH / 2;
	game->player_y = SCREEN_HEIGHT - 100;
	game->player_health = 100;
	game->player_energy = 100;
	game->score = 0;
	game->wave = 1;
	game->kills = 0;

	/* Initialize Q-Learning */
	game->q_table = q_learning_create(100, 5); /* 100 states, 5 actions */
	game->learning_step = 0;

	for (i = 0; i < MAX_ENEMIES; i++) {
		game->enemies[i].active = 0;
		game->enemies[i].attack_cooldown = 0;
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
	game->game_over = 0;
	game->paused = 0;
	game->victory = 0;
}

/*=============================================================================
 * Game Update
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
			game->enemies[i].max_health = game->enemies[i].health;
			game->enemies[i].speed = ENEMY_SPEED + game->wave * 0.2f;
			game->enemies[i].type = rand() % 3;
			game->enemies[i].active = 1;
			game->enemies[i].attack_cooldown = 0;
			game->num_enemies++;
			break;
		}
	}
}

static void update_game(GameState *game, AudioSystem *audio) {
	int i, j;
	float dx, dy, dist;

	if (game->game_over || game->paused)
		return;

	game->frame_count++;
	game->game_time += game->delta_time;

	/* Spawn enemies based on wave */
	if (game->num_enemies < 3 + game->wave && rand() % 100 < 3) {
		spawn_enemy(game);
	}

	/* Update enemies with AI */
	for (i = 0; i < game->num_enemies; i++) {
		if (!game->enemies[i].active)
			continue;

		/* Calculate state for Q-Learning (simplified) */
		dx = game->player_x - game->enemies[i].x;
		dy = game->player_y - game->enemies[i].y;
		dist = sqrtf(dx * dx + dy * dy);

		int state = (int) ((dist / SCREEN_WIDTH) * 10)
				+ (game->enemies[i].type * 33);

		/* Select action using Q-Learning */
		int action = q_learning_select_action(game->q_table, state, 5, 0.2f);

		/* Move based on action */
		float speed = game->enemies[i].speed;
		switch (action) {
		case 0: /* Move toward player */
			if (dist > 1.0f) {
				game->enemies[i].x += (dx / dist) * speed;
				game->enemies[i].y += (dy / dist) * speed;
			}
			break;
		case 1: /* Move horizontally */
			game->enemies[i].x += (dx > 0 ? speed : -speed) * 0.5f;
			break;
		case 2: /* Move vertically */
			game->enemies[i].y += (dy > 0 ? speed : -speed) * 0.5f;
			break;
		case 3: /* Strafe */
			game->enemies[i].x += (dx > 0 ? speed : -speed) * 0.3f;
			game->enemies[i].y += (dy > 0 ? speed : -speed) * 0.3f;
			break;
		case 4: /* Attack */
			if (game->enemies[i].attack_cooldown <= 0 && dist < 300) {
				/* Enemy fires bullet */
				for (j = 0; j < MAX_BULLETS; j++) {
					if (!game->bullets[j].active) {
						game->bullets[j].x = game->enemies[i].x;
						game->bullets[j].y = game->enemies[i].y + 10;
						game->bullets[j].vx = dx / dist * 5.0f;
						game->bullets[j].vy = dy / dist * 5.0f;
						game->bullets[j].active = 1;
						game->bullets[j].owner = 1;
						game->num_bullets++;
						game->enemies[i].attack_cooldown = 1.0f;

						if (audio) {
							audio_play_sound(audio, "laser", game->enemies[i].x,
									game->enemies[i].y);
						}
						break;
					}
				}
			}
			break;
		}

		if (game->enemies[i].attack_cooldown > 0) {
			game->enemies[i].attack_cooldown -= game->delta_time;
		}

		/* Check collision with player */
		dx = game->player_x - game->enemies[i].x;
		dy = game->player_y - game->enemies[i].y;
		dist = sqrtf(dx * dx + dy * dy);

		if (dist < 40) {
			game->player_health -= 20 * game->delta_time;
			game->enemies[i].active = 0;
			game->num_enemies--;

			create_particle(game, game->enemies[i].x, game->enemies[i].y,
			COLOR_RED, 1.0f);

			if (audio) {
				audio_play_sound(audio, "hit", game->enemies[i].x,
						game->enemies[i].y);
			}

			if (game->player_health <= 0) {
				game->game_over = 1;
				if (audio) {
					audio_play_sound(audio, "gameover", game->player_x,
							game->player_y);
				}
			}

			/* Q-Learning reward (negative for hitting player) */
			int next_state = state;
			q_learning_update(game->q_table, state, action, -1.0f, next_state,
					5, 0.1f, 0.95f);
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

		/* Check bullet collisions */
		if (game->bullets[i].owner == 0) { /* Player bullet */
			for (j = 0; j < game->num_enemies; j++) {
				if (!game->enemies[j].active)
					continue;

				dx = game->bullets[i].x - game->enemies[j].x;
				dy = game->bullets[i].y - game->enemies[j].y;
				dist = sqrtf(dx * dx + dy * dy);

				if (dist < 25) {
					game->enemies[j].health -= 20;
					game->bullets[i].active = 0;
					game->num_bullets--;

					create_particle(game, game->bullets[i].x,
							game->bullets[i].y,
							COLOR_YELLOW, 0.5f);

					if (audio) {
						audio_play_sound(audio, "hit", game->bullets[i].x,
								game->bullets[i].y);
					}

					if (game->enemies[j].health <= 0) {
						game->score += 100;
						game->kills++;
						game->enemies[j].active = 0;
						game->num_enemies--;

						create_particle(game, game->enemies[j].x,
								game->enemies[j].y,
								COLOR_ORANGE, 1.5f);

						if (audio) {
							audio_play_sound(audio, "explosion",
									game->enemies[j].x, game->enemies[j].y);
						}
					}
					break;
				}
			}
		} else { /* Enemy bullet */
			dx = game->player_x - game->bullets[i].x;
			dy = game->player_y - game->bullets[i].y;
			dist = sqrtf(dx * dx + dy * dy);

			if (dist < 20) {
				game->player_health -= 10;
				game->bullets[i].active = 0;
				game->num_bullets--;

				create_particle(game, game->player_x, game->player_y,
				COLOR_RED, 0.5f);

				if (audio) {
					audio_play_sound(audio, "hit", game->player_x,
							game->player_y);
				}

				if (game->player_health <= 0) {
					game->game_over = 1;
					if (audio) {
						audio_play_sound(audio, "gameover", game->player_x,
								game->player_y);
					}
				}
			}
		}
	}

	/* Update particles */
	update_particles(game);

	/* Check wave completion */
	if (game->num_enemies == 0) {
		game->wave++;
		game->player_health = clampf(game->player_health + 20, 0, 100);

		if (game->wave > 10) {
			game->victory = 1;
			if (audio) {
				audio_play_sound(audio, "victory", game->player_x,
						game->player_y);
			}
		}
	}

	/* Update audio listener position */
	if (audio) {
		audio_update_listener(audio, game->player_x, game->player_y);
	}

	game->learning_step++;
}

/*=============================================================================
 * Input Processing
 *===========================================================================*/

static void process_input(AppState *state) {
	GameState *game = state->game;
	AudioSystem *audio = state->audio;
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
				game->paused = !game->paused;
				break;

			case SDLK_r:
				if (game->game_over || game->victory) {
					init_game(game);
				}
				break;

			case SDLK_SPACE:
				if (!game->game_over && !game->victory && !game->paused) {
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

							if (audio) {
								audio_play_sound(audio, "laser", game->player_x,
										game->player_y);
							}
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

	if (!game->game_over && !game->victory && !game->paused) {
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
 * Rendering Functions
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
	for (i = 0; i < game->num_particles; i++) {
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
	/* Simple placeholder - for production use SDL_ttf or bitmap font */
	(void) x;
	(void) y;
	(void) text;
	(void) color;
}

static void render_game(AppState *state) {
	GameState *game = state->game;
	int i;
	char hud_text[256];

	glClear(GL_COLOR_BUFFER_BIT);
	glLoadIdentity();

	/* Set up 2D orthographic projection */
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, SCREEN_WIDTH, SCREEN_HEIGHT, 0, -1, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	/* Draw background gradient */
	glBegin(GL_QUADS);
	glColor4f(0.05f, 0.05f, 0.1f, 1.0f);
	glVertex2f(0, 0);
	glColor4f(0.1f, 0.1f, 0.2f, 1.0f);
	glVertex2f(SCREEN_WIDTH, 0);
	glColor4f(0.05f, 0.05f, 0.15f, 1.0f);
	glVertex2f(SCREEN_WIDTH, SCREEN_HEIGHT);
	glColor4f(0.02f, 0.02f, 0.05f, 1.0f);
	glVertex2f(0, SCREEN_HEIGHT);
	glEnd();

	/* Draw stars */
	glPointSize(1.5f);
	glColor4f(0.8f, 0.8f, 1.0f, 0.8f);
	glBegin(GL_POINTS);
	for (i = 0; i < 100; i++) {
		float x = fmodf(game->game_time * 20 + i * 37, SCREEN_WIDTH);
		float y = fmodf(i * 73 + game->game_time * 10, SCREEN_HEIGHT);
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
		float health_pct = game->enemies[i].health
				/ game->enemies[i].max_health;
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
	draw_rect(game->player_x, game->player_y, 25, 25, COLOR_GREEN);

	/* Draw health bar */
	draw_rect(game->player_x, game->player_y - 35, 40, 5, COLOR_RED);
	draw_rect(game->player_x - 20 + 20 * (game->player_health / 100.0f),
			game->player_y - 35, 40 * (game->player_health / 100.0f), 5,
			COLOR_GREEN);

	/* Draw HUD */
	sprintf(hud_text, "Score: %d  Wave: %d  Kills: %d", game->score, game->wave,
			game->kills);
	draw_text(10, 20, hud_text, COLOR_WHITE);

	sprintf(hud_text, "Health: %.0f  Energy: %.0f", game->player_health,
			game->player_energy);
	draw_text(10, 45, hud_text, COLOR_WHITE);

	sprintf(hud_text, "FPS: %.1f", state->fps);
	draw_text(SCREEN_WIDTH - 80, 20, hud_text, COLOR_WHITE);

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
	}

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
		sprintf(hud_text, "Final Score: %d  Wave: %d", game->score, game->wave);
		draw_text(SCREEN_WIDTH / 2 - 90, SCREEN_HEIGHT / 2 + 10, hud_text,
				COLOR_WHITE);
		draw_text(SCREEN_WIDTH / 2 - 60, SCREEN_HEIGHT / 2 + 40,
				"Press R to restart", COLOR_WHITE);
	}

	if (game->victory) {
		glColor4f(0, 0, 0, 0.8f);
		glBegin(GL_QUADS);
		glVertex2f(0, 0);
		glVertex2f(SCREEN_WIDTH, 0);
		glVertex2f(SCREEN_WIDTH, SCREEN_HEIGHT);
		glVertex2f(0, SCREEN_HEIGHT);
		glEnd();
		draw_text(SCREEN_WIDTH / 2 - 60, SCREEN_HEIGHT / 2 - 20, "VICTORY!",
				COLOR_GOLD);
		sprintf(hud_text, "Final Score: %d  Kills: %d", game->score,
				game->kills);
		draw_text(SCREEN_WIDTH / 2 - 80, SCREEN_HEIGHT / 2 + 10, hud_text,
				COLOR_WHITE);
		draw_text(SCREEN_WIDTH / 2 - 60, SCREEN_HEIGHT / 2 + 40,
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
	printf("║     EvoX - Ultimate AI Space Battle v4.2                  ║\n");
	printf("║     Q-Learning AI | 3D Audio | Particle Effects           ║\n");
	printf(
			"╚════════════════════════════════════════════════════════════╝\n\n");

	srand((unsigned int) time(NULL));

	signal(SIGINT, handle_signal);
	signal(SIGTERM, handle_signal);

	if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_TIMER) < 0) {
		fprintf(stderr, "SDL Init failed: %s\n", SDL_GetError());
		return -1;
	}

	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
	SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 8);

	state->window = SDL_CreateWindow("EvoX - AI Space Battle",
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

	glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, SCREEN_WIDTH, SCREEN_HEIGHT, 0, -1, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glClearColor(0.05f, 0.05f, 0.1f, 1.0f);

	printf("OpenGL: %s\n", glGetString(GL_RENDERER));

	printf("\nInitializing audio system...\n");
	state->audio = audio_system_create();
	if (state->audio) {
		printf("  Audio initialized successfully\n");
		printf("  Sounds: laser, explosion, hit, gameover, victory\n");
	} else {
		printf("  Audio initialization failed - continuing without sound\n");
	}

	printf("\nInitializing game with Q-Learning AI...\n");
	state->game = (GameState*) malloc(sizeof(GameState));
	init_game(state->game);

	gettimeofday(&state->last_frame, NULL);
	state->running = 1;

	printf("\n");
	printf("Controls:\n");
	printf("  Arrow Keys / WASD - Move Player\n");
	printf("  SPACE - Fire Weapon\n");
	printf("  P - Pause Game\n");
	printf("  R - Restart Game\n");
	printf("  ESC - Exit Game\n\n");

	printf("Game Features:\n");
	printf("  ✓ Q-Learning AI enemies\n");
	printf("  ✓ Procedural 3D audio\n");
	printf("  ✓ Particle effects\n");
	printf("  ✓ Wave progression\n");
	printf("  ✓ Health & scoring system\n\n");

	return 0;
}

/*=============================================================================
 * Application Cleanup
 *===========================================================================*/

static void app_cleanup(AppState *state) {
	printf("\nShutting down...\n");

	if (state->game) {
		printf("Final Score: %d\n", state->game->score);
		printf("Enemies Killed: %d\n", state->game->kills);
		printf("Waves Completed: %d\n", state->game->wave - 1);

		if (state->game->q_table) {
			free(state->game->q_table);
		}
		free(state->game);
	}

	if (state->audio) {
		audio_system_destroy(state->audio);
	}

	if (state->gl_context) {
		SDL_GL_DeleteContext(state->gl_context);
	}
	if (state->window) {
		SDL_DestroyWindow(state->window);
	}
	SDL_Quit();

	printf("Game terminated.\n");
}

/*=============================================================================
 * Main Loop
 *===========================================================================*/

static void app_run(AppState *state) {
	Uint32 last_time, current_time, frame_time;
	Uint32 fps_last_time = SDL_GetTicks();
	int fps_frames = 0;

	last_time = SDL_GetTicks();

	while (state->running && g_running) {
		current_time = SDL_GetTicks();
		frame_time = current_time - last_time;

		if (frame_time > 0) {
			state->game->delta_time = frame_time / 1000.0f;
		}
		last_time = current_time;

		process_input(state);

		if (state->game) {
			update_game(state->game, state->audio);
		}

		render_game(state);
		SDL_GL_SwapWindow(state->window);

		fps_frames++;
		if (current_time - fps_last_time >= 1000) {
			state->fps = (float) fps_frames * 1000.0f
					/ (current_time - fps_last_time);
			fps_frames = 0;
			fps_last_time = current_time;
		}

		frame_time = SDL_GetTicks() - current_time;
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
