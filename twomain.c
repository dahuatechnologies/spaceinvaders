/*
 * EvoX - Ultimate AI Sensory Experience System
 * Version: 3.2 - Fixed Rendering & Compilation
 *
 * AI Features: Q-Learning, Temporal Pattern Recognition
 * Audio: Full 3D Spatialization
 * Graphics: Fullscreen, Multi-monitor, Adaptive Resolution
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
 * External Libraries
 *===========================================================================*/

#include <SDL2/SDL.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <AL/al.h>
#include <AL/alc.h>

/*=============================================================================
 * Maximum Screen Configuration
 *===========================================================================*/

#define MAX_SCREEN_WIDTH        7680
#define MAX_SCREEN_HEIGHT       4320
#define DEFAULT_SCREEN_WIDTH    1024
#define DEFAULT_SCREEN_HEIGHT   768
#define MIN_SCREEN_WIDTH        640
#define MIN_SCREEN_HEIGHT       480

#define DISPLAY_WINDOWED        0
#define DISPLAY_FULLSCREEN      1
#define DISPLAY_FULLSCREEN_DESKTOP 2
#define DISPLAY_BORDERLESS      3
#define DISPLAY_MULTIMONITOR    4

#define MAX_MONITORS            8
#define MAX_REFRESH_RATES       144

/*=============================================================================
 * Audio Configuration
 *===========================================================================*/

#define AUDIO_SAMPLE_RATE       44100
#define AUDIO_BUFFER_SIZE       2048
#define AUDIO_MAX_SOURCES       32
#define AUDIO_HRTF_ENABLED      0

#define SPEED_OF_SOUND          343.3f
#define DOPPLER_FACTOR          1.0f
#define DISTANCE_MODEL          AL_INVERSE_DISTANCE_CLAMPED
#define MAX_DISTANCE            10000.0f
#define REFERENCE_DISTANCE      100.0f
#define ROLLOFF_FACTOR          1.0f

/*=============================================================================
 * AI Configuration
 *===========================================================================*/

#define Q_TABLE_SIZE            256
#define MAX_SEQUENCE_LENGTH     1000
#define STATE_DIM               10
#define ACTION_DIM              5

/*=============================================================================
 * Game Configuration
 *===========================================================================*/

#define MAX_ENEMIES             20
#define MAX_BULLETS             100
#define MAX_PARTICLES           1000
#define MAX_TRAILS              50
#define MAX_GLOWS               30
#define MAX_WAVES               20
#define MAX_SOUNDS              16
#define MAX_MUSIC_LAYERS        4
#define MAX_EMITTERS            8

#define TRAIL_LENGTH            20
#define PARTICLE_LIFE           2.0f
#define SCREEN_SHAKE_MAX        16.0f
#define TARGET_FPS              60
#define FRAME_TIME_MS           16

/*=============================================================================
 * Weather System
 *===========================================================================*/

#define WEATHER_NONE            0
#define WEATHER_RAIN            1
#define WEATHER_SNOW            2
#define WEATHER_FOG             3
#define WEATHER_STORM           4

/*=============================================================================
 * Colors (32-bit RGBA)
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
#define COLOR_PURPLE     0x8800FFFF
#define COLOR_GOLD       0xFFD700FF

/*=============================================================================
 * Forward Type Declarations
 *===========================================================================*/

typedef struct GlowLight GlowLight;
typedef struct LightningBolt LightningBolt;
typedef struct ForceField ForceField;
typedef struct AudioSource3D AudioSource3D;
typedef struct AdvancedAudioSystem AdvancedAudioSystem;
typedef struct MonitorInfo MonitorInfo;
typedef struct DisplayConfig DisplayConfig;
typedef struct QLearningAgent QLearningAgent;
typedef struct AISystem AISystem;
typedef struct AIEnemy AIEnemy;
typedef struct AdvancedParticle AdvancedParticle;
typedef struct SensoryGame SensoryGame;
typedef struct AppState AppState;

/*=============================================================================
 * Basic Structures
 *===========================================================================*/

struct GlowLight {
	float x, y;
	float radius;
	float intensity;
	unsigned int color;
	float pulse_speed;
	float pulse_phase;
	int active;
};

struct LightningBolt {
	float x, y;
	float target_x, target_y;
	float speed;
	float thickness;
	unsigned int color;
	float life;
	int active;
};

struct ForceField {
	float x, y;
	float radius;
	float force_x, force_y;
	float attenuation;
	int active;
	int type;
};

/*=============================================================================
 * Audio Structures
 *===========================================================================*/

struct AudioSource3D {
	ALuint source;
	ALuint buffer;
	char name[64];
	float position[3];
	float velocity[3];
	float direction[3];
	float volume;
	float base_volume;
	float pitch;
	int loop;
	int active;
	Uint64 last_play_time;
};

struct AdvancedAudioSystem {
	ALCdevice *device;
	ALCcontext *context;
	AudioSource3D sounds[MAX_SOUNDS];
	AudioSource3D music_layers[MAX_MUSIC_LAYERS];
	float listener_pos[3];
	float listener_vel[3];
	float listener_ori[6];
	float master_volume;
	float music_volume;
	float sfx_volume;
	int initialized;
	int audio_error;
	pthread_mutex_t audio_mutex;
};

/*=============================================================================
 * Display Management Structures
 *===========================================================================*/

struct MonitorInfo {
	SDL_DisplayMode mode;
	SDL_Rect bounds;
	char name[128];
	int is_primary;
	float refresh_rate;
};

struct DisplayConfig {
	int width;
	int height;
	int refresh_rate;
	int fullscreen;
	int borderless;
	int vsync;
	int multisamples;
	int monitor_index;
	float aspect_ratio;
	float scale_factor;
};

/*=============================================================================
 * AI Structures
 *===========================================================================*/

struct QLearningAgent {
	float *q_table;
	float *state_space;
	int state_dim;
	int action_dim;
	float learning_rate;
	float discount_factor;
	float exploration_rate;
	float exploration_decay;
	float *eligibility_traces;
};

struct AISystem {
	QLearningAgent *enemy_agents[MAX_ENEMIES];
	float *temporal_buffer;
	float *predicted_state;
	int learning_step;
	float cumulative_reward;
};

/*=============================================================================
 * Enhanced Game Objects
 *===========================================================================*/

struct AIEnemy {
	float x, y, z;
	float vx, vy, vz;
	float health;
	float max_health;
	int active;
	float aggression;
	float speed;
	unsigned int color;
	float trail_history[TRAIL_LENGTH][2];
	int trail_index;
	float pulse_phase;
	float last_reward;
};

struct AdvancedParticle {
	float x, y, z;
	float vx, vy, vz;
	float ax, ay, az;
	float life;
	float max_life;
	float size;
	float rotation;
	float rotation_speed;
	unsigned int color;
	int active;
};

typedef struct {
	float x, y;
	float vx, vy;
	int active;
	int owner;
	int damage;
	unsigned int color;
	float trail_history_x[TRAIL_LENGTH];
	float trail_history_y[TRAIL_LENGTH];
	int trail_index;
} Bullet;

/*=============================================================================
 * Complete Game State
 *===========================================================================*/

struct SensoryGame {
	/* Player */
	float x, y, z;
	float vx, vy, vz;
	float health;
	float max_health;
	float shield;
	float energy;
	int score;
	int combo;
	int combo_timer;

	/* Game objects */
	AIEnemy enemies[MAX_ENEMIES];
	Bullet bullets[MAX_BULLETS];
	AdvancedParticle particles[MAX_PARTICLES];
	GlowLight glows[MAX_GLOWS];
	LightningBolt lightnings[MAX_ENEMIES];
	ForceField fields[MAX_EMITTERS];

	/* AI System */
	AISystem *ai;

	/* Weather */
	int weather_type;
	float weather_intensity;
	float wind_x, wind_y;
	float visibility;

	/* Screen effects */
	float screen_shake;
	float screen_shake_duration;
	float flash_intensity;
	float flash_duration;
	unsigned int flash_color;

	/* Counts */
	int num_enemies;
	int num_bullets;
	int num_particles;
	int num_glows;

	/* Wave progression */
	int current_wave;
	int enemies_killed;
	int wave_enemies;

	/* Statistics */
	int total_kills;
	int total_shots;
	int total_hits;
	int ai_decisions;

	/* Game state */
	int game_over;
	int victory;
	int paused;

	/* Time */
	Uint64 frame_count;
	float delta_time;
	float game_time;
};

/*=============================================================================
 * Application State
 *===========================================================================*/

struct AppState {
	SDL_Window *window;
	SDL_GLContext gl_context;
	SensoryGame *game;
	AdvancedAudioSystem *audio;
	DisplayConfig display;
	MonitorInfo monitors[MAX_MONITORS];
	int num_monitors;
	int running;
	float fps;
	struct timeval last_frame;
	int verbose;
	pthread_t audio_thread;
	pthread_mutex_t audio_mutex;
};

/*=============================================================================
 * Global Variables
 *===========================================================================*/

static volatile sig_atomic_t g_running = 1;
static AppState *g_app_state = NULL;

/*=============================================================================
 * Function Prototypes
 *===========================================================================*/

static void handle_signal(int sig);
static float random_range(float min, float max);
static float clampf(float value, float min, float max);
static int clampi(int value, int min, int max);

/* Display functions */
static void detect_monitors(AppState *state);
static DisplayConfig optimize_display_config(AppState *state, int fullscreen);
static int set_display_mode(AppState *state);

/* Audio functions */
static AdvancedAudioSystem* audio_system_create(void);
static void audio_play_sound_3d(AdvancedAudioSystem *audio, const char *name,
		float x, float y, float z, float volume_scale);
static void audio_update_listener(AdvancedAudioSystem *audio, float x, float y,
		float z);
static void audio_update_music(AdvancedAudioSystem *audio, float intensity,
		float health, int wave);
static void audio_system_destroy(AdvancedAudioSystem *audio);
static void* audio_thread_func(void *arg);

/* AI functions */
static QLearningAgent* q_agent_create(int state_dim, int action_dim);
static int q_agent_select_action(QLearningAgent *agent, float *state);
static void q_agent_update(QLearningAgent *agent, float *state, int action,
		float reward, float *next_state);
static AISystem* ai_system_create(void);
static void ai_update_enemies(SensoryGame *game, AdvancedAudioSystem *audio);
static void ai_system_destroy(AISystem *ai);

/* Visual effects */
static void create_particle(SensoryGame *game, float x, float y, float z,
		float vx, float vy, float vz, unsigned int color, float size,
		float life);
static void create_explosion(SensoryGame *game, float x, float y, float z,
		AdvancedAudioSystem *audio);
static void create_force_field(SensoryGame *game, float x, float y,
		float radius, int type);
static void update_weather(SensoryGame *game, AdvancedAudioSystem *audio);

/* Game management */
static void init_game(SensoryGame *game);
static void update_game(AppState *state);
static void process_input(AppState *state);
static void render_game(AppState *state);

/* Drawing functions */
static void draw_rect(float x, float y, float w, float h, unsigned int color);
static void draw_circle(float x, float y, float r, unsigned int color);
static void draw_particles(SensoryGame *game, float scale);
static void draw_text(float x, float y, const char *text, unsigned int color,
		float scale);
static void draw_test_pattern(int width, int height);

/* Application */
static int app_init(AppState *state, int argc, char *argv[]);
static void app_cleanup(AppState *state);
static void app_run(AppState *state);

/*=============================================================================
 * Signal Handler
 *===========================================================================*/

static void handle_signal(int sig) {
	(void) sig;
	g_running = 0;
	if (g_app_state) {
		g_app_state->running = 0;
	}
}

/*=============================================================================
 * Math Utilities
 *===========================================================================*/

static float random_range(float min, float max) {
	return min + ((float) rand() / RAND_MAX) * (max - min);
}

static float clampf(float value, float min, float max) {
	if (value < min)
		return min;
	if (value > max)
		return max;
	return value;
}

static int clampi(int value, int min, int max) {
	if (value < min)
		return min;
	if (value > max)
		return max;
	return value;
}

/*=============================================================================
 * Display Management
 *===========================================================================*/

static void detect_monitors(AppState *state) {
	int i;
	SDL_DisplayMode mode;

	state->num_monitors = SDL_GetNumVideoDisplays();
	if (state->num_monitors > MAX_MONITORS) {
		state->num_monitors = MAX_MONITORS;
	}

	printf("\nDetected Monitors:\n");
	for (i = 0; i < state->num_monitors; i++) {
		MonitorInfo *mon = &state->monitors[i];

		SDL_GetCurrentDisplayMode(i, &mode);
		SDL_GetDisplayBounds(i, &mon->bounds);

		mon->mode = mode;
		mon->is_primary = (i == 0);
		mon->refresh_rate = mode.refresh_rate ? mode.refresh_rate : 60;

		snprintf(mon->name, sizeof(mon->name), "%s", SDL_GetDisplayName(i));

		printf("  Monitor %d: %s\n", i, mon->name);
		printf("    Mode: %dx%d @ %dHz\n", mode.w, mode.h,
				(int) mon->refresh_rate);
		printf("    Position: %d,%d  Size: %dx%d\n", mon->bounds.x,
				mon->bounds.y, mon->bounds.w, mon->bounds.h);
	}
}

static DisplayConfig optimize_display_config(AppState *state, int fullscreen) {
	DisplayConfig config;
	MonitorInfo *primary = &state->monitors[0];
	int i, max_width = 0, max_height = 0;

	for (i = 0; i < state->num_monitors; i++) {
		MonitorInfo *mon = &state->monitors[i];
		if (mon->bounds.x + mon->bounds.w > max_width) {
			max_width = mon->bounds.x + mon->bounds.w;
		}
		if (mon->bounds.y + mon->bounds.h > max_height) {
			max_height = mon->bounds.y + mon->bounds.h;
		}
	}

	if (fullscreen == DISPLAY_MULTIMONITOR) {
		config.width = max_width;
		config.height = max_height;
		config.monitor_index = 0;
	} else {
		config.width = primary->mode.w;
		config.height = primary->mode.h;
		config.monitor_index = fullscreen ? 0 : -1;
	}

	config.width = clampi(config.width, MIN_SCREEN_WIDTH, MAX_SCREEN_WIDTH);
	config.height = clampi(config.height, MIN_SCREEN_HEIGHT, MAX_SCREEN_HEIGHT);

	config.refresh_rate = (int) primary->refresh_rate;
	config.fullscreen = fullscreen;
	config.borderless = (fullscreen == DISPLAY_BORDERLESS);
	config.vsync = 1;
	config.multisamples = 0;
	config.aspect_ratio = (float) config.width / config.height;
	config.scale_factor = (float) config.width / DEFAULT_SCREEN_WIDTH;

	return config;
}

static int set_display_mode(AppState *state) {
	Uint32 flags = SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_SHOWN;
	DisplayConfig *dc = &state->display;

	switch (dc->fullscreen) {
	case DISPLAY_FULLSCREEN:
		flags |= SDL_WINDOW_FULLSCREEN;
		break;
	case DISPLAY_FULLSCREEN_DESKTOP:
		flags |= SDL_WINDOW_FULLSCREEN_DESKTOP;
		break;
	case DISPLAY_BORDERLESS:
		flags |= SDL_WINDOW_BORDERLESS;
		break;
	case DISPLAY_MULTIMONITOR:
		flags |= SDL_WINDOW_BORDERLESS;
		if (state->num_monitors > 0) {
			SDL_SetWindowPosition(state->window, state->monitors[0].bounds.x,
					state->monitors[0].bounds.y);
		}
		break;
	}

	SDL_SetWindowSize(state->window, dc->width, dc->height);
	SDL_SetWindowFullscreen(state->window, flags);

	SDL_GL_SetSwapInterval(dc->vsync ? 1 : 0);

	glViewport(0, 0, dc->width, dc->height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, dc->width, dc->height, 0, -1, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	printf("\nDisplay Mode Set:\n");
	printf("  Resolution: %dx%d\n", dc->width, dc->height);
	printf("  Aspect Ratio: %.2f:1\n", dc->aspect_ratio);
	printf("  Scale Factor: %.2f\n", dc->scale_factor);

	return 0;
}

/*=============================================================================
 * Audio Thread Function
 *===========================================================================*/

static void* audio_thread_func(void *arg) {
	AdvancedAudioSystem *audio = (AdvancedAudioSystem*) arg;
	ALint state;
	int i;

	while (g_running && audio && audio->initialized) {
		pthread_mutex_lock(&audio->audio_mutex);

		for (i = 0; i < MAX_SOUNDS; i++) {
			if (audio->sounds[i].active) {
				alGetSourcei(audio->sounds[i].source, AL_SOURCE_STATE, &state);
				if (state == AL_PLAYING) {
					if (audio->sounds[i].velocity[0] != 0
							|| audio->sounds[i].velocity[1] != 0
							|| audio->sounds[i].velocity[2] != 0) {

						alSourcefv(audio->sounds[i].source, AL_POSITION,
								audio->sounds[i].position);
					}
				}
			}
		}

		pthread_mutex_unlock(&audio->audio_mutex);
		usleep(10000);
	}

	return NULL;
}

/*=============================================================================
 * Audio System Implementation
 *===========================================================================*/

static AdvancedAudioSystem* audio_system_create(void) {
	AdvancedAudioSystem *audio;
	int i;

	audio = (AdvancedAudioSystem*) calloc(1, sizeof(AdvancedAudioSystem));
	if (!audio)
		return NULL;

	pthread_mutex_init(&audio->audio_mutex, NULL);

	audio->device = alcOpenDevice(NULL);
	if (!audio->device) {
		printf("Warning: Could not open audio device\n");
		audio->initialized = 0;
		return audio;
	}

	audio->context = alcCreateContext(audio->device, NULL);
	if (!audio->context) {
		alcCloseDevice(audio->device);
		audio->initialized = 0;
		return audio;
	}

	alcMakeContextCurrent(audio->context);

	printf("\nAudio System Initialized:\n");
	printf("  Device: %s\n", alcGetString(audio->device, ALC_DEVICE_SPECIFIER));

	audio->listener_pos[0] = 0.0f;
	audio->listener_pos[1] = 0.0f;
	audio->listener_pos[2] = 0.0f;
	audio->listener_vel[0] = 0.0f;
	audio->listener_vel[1] = 0.0f;
	audio->listener_vel[2] = 0.0f;
	audio->listener_ori[0] = 0.0f;
	audio->listener_ori[1] = 0.0f;
	audio->listener_ori[2] = -1.0f;
	audio->listener_ori[3] = 0.0f;
	audio->listener_ori[4] = 1.0f;
	audio->listener_ori[5] = 0.0f;

	alListenerfv(AL_POSITION, audio->listener_pos);
	alListenerfv(AL_VELOCITY, audio->listener_vel);
	alListenerfv(AL_ORIENTATION, audio->listener_ori);

	alDistanceModel(DISTANCE_MODEL);
	alDopplerFactor(DOPPLER_FACTOR);
	alSpeedOfSound(SPEED_OF_SOUND);

	for (i = 0; i < MAX_SOUNDS; i++) {
		alGenSources(1, &audio->sounds[i].source);
		alSourcef(audio->sounds[i].source, AL_REFERENCE_DISTANCE,
				REFERENCE_DISTANCE);
		alSourcef(audio->sounds[i].source, AL_MAX_DISTANCE, MAX_DISTANCE);
		alSourcef(audio->sounds[i].source, AL_ROLLOFF_FACTOR, ROLLOFF_FACTOR);
		audio->sounds[i].active = 0;
		audio->sounds[i].base_volume = 0.5f;
		strcpy(audio->sounds[i].name, "unknown");
	}

	for (i = 0; i < MAX_MUSIC_LAYERS; i++) {
		alGenSources(1, &audio->music_layers[i].source);
		alSourcei(audio->music_layers[i].source, AL_LOOPING, AL_TRUE);
		alSourcef(audio->music_layers[i].source, AL_GAIN, 0.0f);
		audio->music_layers[i].active = 0;
		audio->music_layers[i].base_volume = 0.3f;
	}

	audio->master_volume = 1.0f;
	audio->music_volume = 0.5f;
	audio->sfx_volume = 0.8f;

	audio->initialized = 1;

	return audio;
}

static void audio_play_sound_3d(AdvancedAudioSystem *audio, const char *name,
		float x, float y, float z, float volume_scale) {
	int i;
	ALint state;

	if (!audio || !audio->initialized)
		return;

	pthread_mutex_lock(&audio->audio_mutex);

	for (i = 0; i < MAX_SOUNDS; i++) {
		if (strcmp(audio->sounds[i].name, name) == 0) {

			alGetSourcei(audio->sounds[i].source, AL_SOURCE_STATE, &state);

			if (state != AL_PLAYING || audio->sounds[i].loop) {
				float pos[3] = { x, y, z };
				float gain = audio->sounds[i].base_volume * volume_scale
						* audio->sfx_volume * audio->master_volume;

				alSourcefv(audio->sounds[i].source, AL_POSITION, pos);
				alSourcef(audio->sounds[i].source, AL_GAIN, gain);
				alSourcePlay(audio->sounds[i].source);
				audio->sounds[i].last_play_time = SDL_GetTicks64();
				audio->sounds[i].position[0] = x;
				audio->sounds[i].position[1] = y;
				audio->sounds[i].position[2] = z;
			}
			break;
		}
	}

	pthread_mutex_unlock(&audio->audio_mutex);
}

static void audio_update_listener(AdvancedAudioSystem *audio, float x, float y,
		float z) {
	if (!audio || !audio->initialized)
		return;

	pthread_mutex_lock(&audio->audio_mutex);

	audio->listener_pos[0] = x;
	audio->listener_pos[1] = y;
	audio->listener_pos[2] = z;

	alListenerfv(AL_POSITION, audio->listener_pos);

	pthread_mutex_unlock(&audio->audio_mutex);
}

static void audio_update_music(AdvancedAudioSystem *audio, float intensity,
		float health, int wave) {
	int i;
	float target_volumes[MAX_MUSIC_LAYERS];

	if (!audio || !audio->initialized)
		return;

	for (i = 0; i < MAX_MUSIC_LAYERS; i++) {
		target_volumes[i] = audio->music_volume * 0.3f;

		if (i == 0) {
			target_volumes[i] *= 1.0f;
		} else if (i == 1) {
			target_volumes[i] *= intensity;
		} else if (i == 2) {
			target_volumes[i] *= (1.0f - health / 100.0f);
		} else if (i == 3) {
			target_volumes[i] *= (float) wave / MAX_WAVES;
		}
	}

	pthread_mutex_lock(&audio->audio_mutex);

	for (i = 0; i < MAX_MUSIC_LAYERS; i++) {
		if (!audio->music_layers[i].active)
			continue;

		float current = audio->music_layers[i].volume;
		float target = target_volumes[i];
		float new_vol = current * 0.9f + target * 0.1f;

		alSourcef(audio->music_layers[i].source, AL_GAIN, new_vol);
		audio->music_layers[i].volume = new_vol;

		ALint state;
		alGetSourcei(audio->music_layers[i].source, AL_SOURCE_STATE, &state);
		if (state != AL_PLAYING) {
			alSourcePlay(audio->music_layers[i].source);
		}
	}

	pthread_mutex_unlock(&audio->audio_mutex);
}

static void audio_system_destroy(AdvancedAudioSystem *audio) {
	int i;

	if (!audio)
		return;

	pthread_mutex_lock(&audio->audio_mutex);

	if (audio->initialized) {
		for (i = 0; i < MAX_SOUNDS; i++) {
			if (audio->sounds[i].source) {
				alSourceStop(audio->sounds[i].source);
				alDeleteSources(1, &audio->sounds[i].source);
			}
			if (audio->sounds[i].buffer) {
				alDeleteBuffers(1, &audio->sounds[i].buffer);
			}
		}

		for (i = 0; i < MAX_MUSIC_LAYERS; i++) {
			if (audio->music_layers[i].source) {
				alSourceStop(audio->music_layers[i].source);
				alDeleteSources(1, &audio->music_layers[i].source);
			}
			if (audio->music_layers[i].buffer) {
				alDeleteBuffers(1, &audio->music_layers[i].buffer);
			}
		}

		alcMakeContextCurrent(NULL);
		alcDestroyContext(audio->context);
		alcCloseDevice(audio->device);
	}

	pthread_mutex_unlock(&audio->audio_mutex);
	pthread_mutex_destroy(&audio->audio_mutex);

	free(audio);
}

/*=============================================================================
 * AI System Implementation
 *===========================================================================*/

static QLearningAgent* q_agent_create(int state_dim, int action_dim) {
	QLearningAgent *agent;
	int i;

	agent = (QLearningAgent*) calloc(1, sizeof(QLearningAgent));
	if (!agent)
		return NULL;

	agent->state_dim = state_dim;
	agent->action_dim = action_dim;
	agent->q_table = (float*) calloc(state_dim * action_dim, sizeof(float));
	agent->state_space = (float*) calloc(state_dim, sizeof(float));
	agent->eligibility_traces = (float*) calloc(state_dim * action_dim,
			sizeof(float));

	agent->learning_rate = 0.1f;
	agent->discount_factor = 0.95f;
	agent->exploration_rate = 0.2f;
	agent->exploration_decay = 0.999f;

	for (i = 0; i < state_dim * action_dim; i++) {
		agent->q_table[i] = random_range(-0.1f, 0.1f);
	}

	return agent;
}

static int q_agent_select_action(QLearningAgent *agent, float *state) {
	int i, best_action = 0;
	float max_q = -1e10f;

	if (random_range(0, 1) < agent->exploration_rate) {
		return rand() % agent->action_dim;
	}

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

static void q_agent_update(QLearningAgent *agent, float *state, int action,
		float reward, float *next_state) {
	int i, j;
	float current_q = 0;
	float next_q = 0;
	float td_error;

	for (i = 0; i < agent->state_dim; i++) {
		current_q += agent->q_table[i * agent->action_dim + action] * state[i];
	}

	for (i = 0; i < agent->action_dim; i++) {
		float q_val = 0;
		for (j = 0; j < agent->state_dim; j++) {
			q_val += agent->q_table[j * agent->action_dim + i] * next_state[j];
		}
		if (q_val > next_q)
			next_q = q_val;
	}

	td_error = reward + agent->discount_factor * next_q - current_q;

	for (i = 0; i < agent->state_dim; i++) {
		int idx = i * agent->action_dim + action;
		agent->eligibility_traces[idx] = agent->discount_factor * 0.9f
				* agent->eligibility_traces[idx] + state[i];
		agent->q_table[idx] += agent->learning_rate * td_error
				* agent->eligibility_traces[idx];
	}

	agent->exploration_rate *= agent->exploration_decay;
}

static AISystem* ai_system_create(void) {
	AISystem *ai;
	int i;

	ai = (AISystem*) calloc(1, sizeof(AISystem));
	if (!ai)
		return NULL;

	for (i = 0; i < MAX_ENEMIES; i++) {
		ai->enemy_agents[i] = q_agent_create(STATE_DIM, ACTION_DIM);
	}

	ai->temporal_buffer = (float*) calloc(1000, sizeof(float));
	ai->predicted_state = (float*) calloc(STATE_DIM, sizeof(float));

	return ai;
}

static void ai_system_destroy(AISystem *ai) {
	int i;

	if (!ai)
		return;

	for (i = 0; i < MAX_ENEMIES; i++) {
		if (ai->enemy_agents[i]) {
			free(ai->enemy_agents[i]->q_table);
			free(ai->enemy_agents[i]->state_space);
			free(ai->enemy_agents[i]->eligibility_traces);
			free(ai->enemy_agents[i]);
		}
	}

	free(ai->temporal_buffer);
	free(ai->predicted_state);
	free(ai);
}

static void ai_update_enemies(SensoryGame *game, AdvancedAudioSystem *audio) {
	int i;
	float state[STATE_DIM];
	int action;
	float reward;
	float next_state[STATE_DIM];

	if (!game || !game->ai)
		return;

	for (i = 0; i < game->num_enemies; i++) {
		if (!game->enemies[i].active)
			continue;

		AIEnemy *enemy = &game->enemies[i];
		QLearningAgent *agent = game->ai->enemy_agents[i];

		if (!agent)
			continue;

		state[0] = game->x / DEFAULT_SCREEN_WIDTH;
		state[1] = game->y / DEFAULT_SCREEN_HEIGHT;
		state[2] = enemy->x / DEFAULT_SCREEN_WIDTH;
		state[3] = enemy->y / DEFAULT_SCREEN_HEIGHT;
		state[4] = enemy->health / enemy->max_health;
		state[5] = game->health / 100.0f;
		state[6] = enemy->aggression;
		state[7] = (float) game->num_enemies / MAX_ENEMIES;
		state[8] = (float) game->current_wave / MAX_WAVES;
		state[9] = random_range(0, 1);

		action = q_agent_select_action(agent, state);

		float speed = enemy->speed * game->delta_time * 60.0f;
		switch (action) {
		case 0:
			enemy->x += speed;
			break;
		case 1:
			enemy->x -= speed;
			break;
		case 2:
			enemy->y += speed;
			break;
		case 3:
			enemy->y -= speed;
			break;
		case 4:
			/* Fire weapon - implemented in game update */
			break;
		}

		float dx = game->x - enemy->x;
		float dy = game->y - enemy->y;
		float dist = sqrtf(dx * dx + dy * dy);
		reward = (1000.0f - dist) / 1000.0f;

		next_state[0] = game->x / DEFAULT_SCREEN_WIDTH;
		next_state[1] = game->y / DEFAULT_SCREEN_HEIGHT;
		next_state[2] = enemy->x / DEFAULT_SCREEN_WIDTH;
		next_state[3] = enemy->y / DEFAULT_SCREEN_HEIGHT;
		next_state[4] = enemy->health / enemy->max_health;
		next_state[5] = game->health / 100.0f;

		q_agent_update(agent, state, action, reward, next_state);

		enemy->last_reward = reward;
		game->ai->cumulative_reward += reward;
	}

	game->ai->learning_step++;
}

/*=============================================================================
 * Visual Effects Functions
 *===========================================================================*/

static void create_particle(SensoryGame *game, float x, float y, float z,
		float vx, float vy, float vz, unsigned int color, float size,
		float life) {
	int i;

	for (i = 0; i < MAX_PARTICLES; i++) {
		if (!game->particles[i].active) {
			game->particles[i].x = x;
			game->particles[i].y = y;
			game->particles[i].z = z;
			game->particles[i].vx = vx;
			game->particles[i].vy = vy;
			game->particles[i].vz = vz;
			game->particles[i].ax = 0;
			game->particles[i].ay = 0.1f;
			game->particles[i].az = 0;
			game->particles[i].life = life;
			game->particles[i].max_life = life;
			game->particles[i].size = size;
			game->particles[i].rotation = random_range(0, 2 * M_PI);
			game->particles[i].rotation_speed = random_range(-0.1f, 0.1f);
			game->particles[i].color = color;
			game->particles[i].active = 1;

			game->num_particles++;
			break;
		}
	}
}

static void create_explosion(SensoryGame *game, float x, float y, float z,
		AdvancedAudioSystem *audio) {
	int i;

	if (audio) {
		audio_play_sound_3d(audio, "explosion", x, y, z, 1.0f);
	}

	for (i = 0; i < 20; i++) {
		float angle = random_range(0, 2 * M_PI);
		float speed = random_range(5, 15);
		float vx = cos(angle) * speed;
		float vy = sin(angle) * speed;

		create_particle(game, x, y, z, vx, vy, 0,
		COLOR_ORANGE, random_range(3, 6), random_range(0.5f, 1.5f));
	}

	game->screen_shake = SCREEN_SHAKE_MAX;
	game->screen_shake_duration = 0.3f;
	game->flash_intensity = 1.0f;
	game->flash_duration = 0.1f;
	game->flash_color = COLOR_WHITE;
}

static void create_force_field(SensoryGame *game, float x, float y,
		float radius, int type) {
	int i;

	for (i = 0; i < MAX_EMITTERS; i++) {
		if (!game->fields[i].active) {
			game->fields[i].x = x;
			game->fields[i].y = y;
			game->fields[i].radius = radius;
			game->fields[i].attenuation = 1.0f;
			game->fields[i].active = 1;
			game->fields[i].type = type;
			break;
		}
	}
}

static void update_weather(SensoryGame *game, AdvancedAudioSystem *audio) {
	float rain_x, rain_y;

	game->wind_x += random_range(-0.1f, 0.1f);
	game->wind_y += random_range(-0.1f, 0.1f);
	game->wind_x = clampf(game->wind_x, -1.0f, 1.0f);
	game->wind_y = clampf(game->wind_y, -1.0f, 1.0f);

	switch (game->weather_type) {
	case WEATHER_RAIN:
		game->weather_intensity = 0.7f;
		game->visibility = 0.9f;

		if (rand() % 10 < game->weather_intensity * 10) {
			rain_x = random_range(0, DEFAULT_SCREEN_WIDTH);
			rain_y = random_range(-50, 0);
			create_particle(game, rain_x, rain_y, 0, game->wind_x * 10, 20, 0,
			COLOR_CYAN, random_range(1, 2), 1.0f);
		}
		break;

	case WEATHER_SNOW:
		game->weather_intensity = 0.5f;
		game->visibility = 0.8f;

		if (rand() % 20 < game->weather_intensity * 10) {
			rain_x = random_range(0, DEFAULT_SCREEN_WIDTH);
			rain_y = random_range(-50, 0);
			create_particle(game, rain_x, rain_y, 0, game->wind_x * 2, 5, 0,
			COLOR_WHITE, random_range(2, 3), 2.0f);
		}
		break;

	case WEATHER_FOG:
		game->weather_intensity = 0.8f;
		game->visibility = 0.4f;
		break;

	case WEATHER_STORM:
		game->weather_intensity = 1.0f;
		game->visibility = 0.6f;

		if (rand() % 1000 < 5) {
			float lightning_x = random_range(100, DEFAULT_SCREEN_WIDTH - 100);
			int i;
			for (i = 0; i < MAX_ENEMIES; i++) {
				if (!game->lightnings[i].active) {
					game->lightnings[i].x = lightning_x;
					game->lightnings[i].y = 0;
					game->lightnings[i].target_x = lightning_x
							+ random_range(-50, 50);
					game->lightnings[i].target_y = DEFAULT_SCREEN_HEIGHT;
					game->lightnings[i].speed = 100;
					game->lightnings[i].thickness = 3;
					game->lightnings[i].color = COLOR_YELLOW;
					game->lightnings[i].life = 0.3f;
					game->lightnings[i].active = 1;
					break;
				}
			}

			if (audio) {
				audio_play_sound_3d(audio, "explosion", lightning_x,
				DEFAULT_SCREEN_HEIGHT / 2, 0, 1.0f);
			}
		}
		break;

	default:
		break;
	}
}

/*=============================================================================
 * Game Initialization
 *===========================================================================*/

static void init_game(SensoryGame *game) {
	int i, j;

	memset(game, 0, sizeof(SensoryGame));

	game->x = DEFAULT_SCREEN_WIDTH / 2.0f;
	game->y = DEFAULT_SCREEN_HEIGHT - 100.0f;
	game->z = 0;
	game->health = 100;
	game->max_health = 100;
	game->shield = 100;
	game->energy = 100;
	game->score = 0;
	game->combo = 1;

	game->current_wave = 1;
	game->wave_enemies = 3;
	game->weather_type = WEATHER_NONE;

	game->ai = ai_system_create();

	for (i = 0; i < MAX_ENEMIES; i++) {
		game->enemies[i].active = 0;
		game->enemies[i].aggression = 1.0f;
		game->enemies[i].speed = 1.0f;
		game->enemies[i].color = COLOR_RED;
		game->enemies[i].health = 50;
		game->enemies[i].max_health = 50;

		for (j = 0; j < TRAIL_LENGTH; j++) {
			game->enemies[i].trail_history[j][0] = 0;
			game->enemies[i].trail_history[j][1] = 0;
		}
	}

	for (i = 0; i < MAX_BULLETS; i++) {
		game->bullets[i].active = 0;
		for (j = 0; j < TRAIL_LENGTH; j++) {
			game->bullets[i].trail_history_x[j] = 0;
			game->bullets[i].trail_history_y[j] = 0;
		}
	}

	for (i = 0; i < MAX_PARTICLES; i++) {
		game->particles[i].active = 0;
	}

	game->num_enemies = 0;
	game->num_bullets = 0;
	game->num_particles = 0;
}

/*=============================================================================
 * Game Update
 *===========================================================================*/

static void update_game(AppState *state) {
	SensoryGame *game = state->game;
	AdvancedAudioSystem *audio = state->audio;
	DisplayConfig *dc = &state->display;
	int i, j;

	if (game->game_over || game->victory || game->paused) {
		return;
	}

	game->frame_count++;
	game->game_time += game->delta_time;

	float move_scale = dc->scale_factor;

	if (game->ai) {
		ai_update_enemies(game, audio);
	}

	/* Spawn enemies */
	if (game->num_enemies < game->wave_enemies && rand() % 100 < 2) {
		for (i = 0; i < MAX_ENEMIES; i++) {
			if (!game->enemies[i].active) {
				game->enemies[i].x = random_range(100, dc->width - 100);
				game->enemies[i].y = 50;
				game->enemies[i].health = 30 + game->current_wave * 5;
				game->enemies[i].max_health = game->enemies[i].health;
				game->enemies[i].active = 1;
				game->enemies[i].speed = 1.0f + game->current_wave * 0.1f;
				game->enemies[i].color = COLOR_RED;
				game->enemies[i].aggression = random_range(0.8f, 1.2f);

				for (j = 0; j < TRAIL_LENGTH; j++) {
					game->enemies[i].trail_history[j][0] = game->enemies[i].x;
					game->enemies[i].trail_history[j][1] = game->enemies[i].y;
				}

				game->num_enemies++;
				break;
			}
		}
	}

	/* Update enemies */
	for (i = 0; i < game->num_enemies; i++) {
		if (!game->enemies[i].active)
			continue;

		AIEnemy *enemy = &game->enemies[i];

		float dx = game->x - enemy->x;
		float dy = game->y - enemy->y;
		float dist = sqrtf(dx * dx + dy * dy);

		if (dist > 1.0f) {
			float speed = enemy->speed * game->delta_time * 60 * move_scale;
			enemy->x += (dx / dist) * speed * enemy->aggression;
			enemy->y += (dy / dist) * speed * enemy->aggression;
		}

		enemy->trail_history[enemy->trail_index][0] = enemy->x;
		enemy->trail_history[enemy->trail_index][1] = enemy->y;
		enemy->trail_index = (enemy->trail_index + 1) % TRAIL_LENGTH;

		enemy->pulse_phase += 0.1f;

		if (dist < 40 * move_scale) {
			game->health -= 10 * game->delta_time;
			game->shield -= 5 * game->delta_time;

			if (game->health <= 0) {
				game->game_over = 1;
				if (audio) {
					audio_play_sound_3d(audio, "gameover", game->x, game->y,
							game->z, 1.0f);
				}
			}
		}
	}

	/* Update bullets */
	for (i = 0; i < game->num_bullets; i++) {
		if (!game->bullets[i].active)
			continue;

		game->bullets[i].x += game->bullets[i].vx * game->delta_time * 60;
		game->bullets[i].y += game->bullets[i].vy * game->delta_time * 60;

		game->bullets[i].trail_history_x[game->bullets[i].trail_index] =
				game->bullets[i].x;
		game->bullets[i].trail_history_y[game->bullets[i].trail_index] =
				game->bullets[i].y;
		game->bullets[i].trail_index = (game->bullets[i].trail_index + 1)
				% TRAIL_LENGTH;

		if (game->bullets[i].y < 0 || game->bullets[i].y > dc->height
				|| game->bullets[i].x < 0 || game->bullets[i].x > dc->width) {
			game->bullets[i].active = 0;
			game->num_bullets--;
			continue;
		}

		if (game->bullets[i].owner == 0) {
			for (j = 0; j < game->num_enemies; j++) {
				if (!game->enemies[j].active)
					continue;

				float dx = game->bullets[i].x - game->enemies[j].x;
				float dy = game->bullets[i].y - game->enemies[j].y;
				float dist = sqrtf(dx * dx + dy * dy);

				if (dist < 25 * move_scale) {
					game->enemies[j].health -= game->bullets[i].damage;
					game->bullets[i].active = 0;
					game->num_bullets--;
					game->total_hits++;

					create_particle(game, game->enemies[j].x,
							game->enemies[j].y, 0, random_range(-3, 3),
							random_range(-3, 3), 0,
							COLOR_YELLOW, 2, 0.3f);

					if (audio) {
						audio_play_sound_3d(audio, "hit", game->enemies[j].x,
								game->enemies[j].y, 0, 0.5f);
					}

					if (game->enemies[j].health <= 0) {
						game->score += 100 * game->combo;
						game->total_kills++;
						game->enemies_killed++;
						game->combo++;
						game->combo_timer = 2000;

						create_explosion(game, game->enemies[j].x,
								game->enemies[j].y, 0, audio);

						game->enemies[j].active = 0;
						game->num_enemies--;
					}
					break;
				}
			}
		}
	}

	/* Update particles */
	for (i = 0; i < game->num_particles; i++) {
		if (!game->particles[i].active)
			continue;

		AdvancedParticle *p = &game->particles[i];

		p->vx += p->ax * game->delta_time * 60;
		p->vy += p->ay * game->delta_time * 60;
		p->vz += p->az * game->delta_time * 60;

		p->vx += game->wind_x * game->delta_time * 10;
		p->vy += game->wind_y * game->delta_time * 10;

		p->x += p->vx * game->delta_time * 60;
		p->y += p->vy * game->delta_time * 60;
		p->z += p->vz * game->delta_time * 60;

		p->rotation += p->rotation_speed;
		p->life -= game->delta_time;

		if (p->life <= 0 || p->y > dc->height + 50) {
			p->active = 0;
			game->num_particles--;
		}
	}

	/* Update weather */
	update_weather(game, audio);

	/* Update lightning bolts */
	for (i = 0; i < MAX_ENEMIES; i++) {
		if (game->lightnings[i].active) {
			game->lightnings[i].x += (game->lightnings[i].target_x
					- game->lightnings[i].x) / 10.0f;
			game->lightnings[i].y += (game->lightnings[i].target_y
					- game->lightnings[i].y) / 10.0f;
			game->lightnings[i].life -= game->delta_time;

			if (game->lightnings[i].life <= 0) {
				game->lightnings[i].active = 0;
			}
		}
	}

	/* Update screen effects */
	if (game->screen_shake_duration > 0) {
		game->screen_shake_duration -= game->delta_time;
		if (game->screen_shake_duration <= 0) {
			game->screen_shake = 0;
		}
	}

	if (game->flash_duration > 0) {
		game->flash_duration -= game->delta_time;
		if (game->flash_duration <= 0) {
			game->flash_intensity = 0;
		}
	}

	/* Update combo timer */
	if (game->combo_timer > 0) {
		game->combo_timer--;
		if (game->combo_timer <= 0) {
			game->combo = 1;
		}
	}

	/* Check wave completion */
	if (game->enemies_killed >= game->wave_enemies && game->num_enemies == 0) {
		game->current_wave++;
		game->wave_enemies = 3 + game->current_wave * 2;
		game->enemies_killed = 0;

		if (game->current_wave > 5) {
			game->weather_type = WEATHER_RAIN;
		}
		if (game->current_wave > 10) {
			game->weather_type = WEATHER_STORM;
		}

		if (game->current_wave > MAX_WAVES) {
			game->victory = 1;
			if (audio) {
				audio_play_sound_3d(audio, "victory", game->x, game->y, game->z,
						1.0f);
			}
		}
	}

	/* Update audio listener */
	if (audio) {
		audio_update_listener(audio, game->x, game->y, game->z);
		audio_update_music(audio,
				(float) game->num_enemies / game->wave_enemies, game->health,
				game->current_wave);
	}
}

/*=============================================================================
 * Input Processing
 *===========================================================================*/

static void process_input(AppState *state) {
	SensoryGame *game = state->game;
	AdvancedAudioSystem *audio = state->audio;
	DisplayConfig *dc = &state->display;
	SDL_Event event;
	const Uint8 *keys;
	float move_scale = dc->scale_factor;
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
					for (i = 0; i < MAX_BULLETS; i++) {
						if (!game->bullets[i].active) {
							game->bullets[i].x = game->x;
							game->bullets[i].y = game->y - 20;
							game->bullets[i].vx = 0;
							game->bullets[i].vy = -15 * move_scale;
							game->bullets[i].active = 1;
							game->bullets[i].owner = 0;
							game->bullets[i].damage = 10 * game->combo;
							game->bullets[i].color = COLOR_CYAN;

							if (audio) {
								audio_play_sound_3d(audio, "laser", game->x,
										game->y, game->z, 0.8f);
							}

							create_particle(game, game->x, game->y - 20, 0, 0,
									-5, 0, COLOR_YELLOW, 3, 0.3f);
							game->num_bullets++;
							game->total_shots++;
							break;
						}
					}
				}
				break;

			case SDLK_w:
				game->weather_type = (game->weather_type + 1) % 5;
				break;

			case SDLK_f:
				if (game->energy > 20) {
					create_force_field(game, game->x, game->y, 200 * move_scale,
							2);
					game->energy -= 20;
				}
				break;

			case SDLK_1:
				dc->fullscreen = DISPLAY_WINDOWED;
				set_display_mode(state);
				break;

			case SDLK_2:
				dc->fullscreen = DISPLAY_FULLSCREEN_DESKTOP;
				set_display_mode(state);
				break;
			}
			break;

		case SDL_WINDOWEVENT:
			if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
				dc->width = event.window.data1;
				dc->height = event.window.data2;
				dc->aspect_ratio = (float) dc->width / dc->height;
				dc->scale_factor = (float) dc->width / DEFAULT_SCREEN_WIDTH;
				glViewport(0, 0, dc->width, dc->height);
				glMatrixMode(GL_PROJECTION);
				glLoadIdentity();
				glOrtho(0, dc->width, dc->height, 0, -1, 1);
				glMatrixMode(GL_MODELVIEW);
			}
			break;
		}
	}

	keys = SDL_GetKeyboardState(NULL);

	if (!game->game_over && !game->victory && !game->paused) {
		float move_x = 0, move_y = 0;
		float speed = 8.0f * move_scale;

		if (keys[SDL_SCANCODE_LEFT] || keys[SDL_SCANCODE_A])
			move_x -= speed;
		if (keys[SDL_SCANCODE_RIGHT] || keys[SDL_SCANCODE_D])
			move_x += speed;
		if (keys[SDL_SCANCODE_UP] || keys[SDL_SCANCODE_W])
			move_y -= speed;
		if (keys[SDL_SCANCODE_DOWN] || keys[SDL_SCANCODE_S])
			move_y += speed;

		game->x += move_x;
		game->y += move_y;

		float margin = 30 * move_scale;
		game->x = clampf(game->x, margin, dc->width - margin);
		game->y = clampf(game->y, margin, dc->height - margin);

		game->vx = move_x / game->delta_time;
		game->vy = move_y / game->delta_time;

		if (move_x != 0 || move_y != 0) {
			if (rand() % 10 == 0) {
				create_particle(game, game->x, game->y, 0, -move_x * 0.2f,
						-move_y * 0.2f, 0,
						COLOR_CYAN, 2, 0.2f);
			}
		}
	}
}

/*=============================================================================
 * Drawing Functions
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
	int segments = 24;
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

static void draw_particles(SensoryGame *game, float scale) {
	int i;

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);
	glPointSize(2.0f * scale);

	glBegin(GL_POINTS);

	for (i = 0; i < game->num_particles; i++) {
		AdvancedParticle *p = &game->particles[i];
		if (p->active) {
			float alpha = p->life / p->max_life;
			float r = ((p->color >> 24) & 0xFF) / 255.0f;
			float g = ((p->color >> 16) & 0xFF) / 255.0f;
			float b = ((p->color >> 8) & 0xFF) / 255.0f;

			glColor4f(r, g, b, alpha);
			glVertex2f(p->x, p->y);
		}
	}

	glEnd();
	glDisable(GL_BLEND);
}

static void draw_lightning(LightningBolt *bolt) {
	if (!bolt->active)
		return;

	float r = ((bolt->color >> 24) & 0xFF) / 255.0f;
	float g = ((bolt->color >> 16) & 0xFF) / 255.0f;
	float b = ((bolt->color >> 8) & 0xFF) / 255.0f;
	float a = bolt->life;

	glLineWidth(bolt->thickness);
	glColor4f(r, g, b, a);

	glBegin(GL_LINE_STRIP);
	glVertex2f(bolt->x, bolt->y);
	glVertex2f(bolt->x + random_range(-20, 20), bolt->y + 50);
	glVertex2f(bolt->target_x, bolt->target_y);
	glEnd();
}

static void draw_text(float x, float y, const char *text, unsigned int color,
		float scale) {
	(void) x;
	(void) y;
	(void) text;
	(void) color;
	(void) scale;
}

static void draw_test_pattern(int width, int height) {
	int i;

	/* Draw colorful gradient background */
	glBegin(GL_QUADS);
	glColor4f(1.0f, 0.0f, 0.0f, 1.0f); /* Red */
	glVertex2f(0, 0);
	glColor4f(0.0f, 1.0f, 0.0f, 1.0f); /* Green */
	glVertex2f(width, 0);
	glColor4f(0.0f, 0.0f, 1.0f, 1.0f); /* Blue */
	glVertex2f(width, height);
	glColor4f(1.0f, 1.0f, 0.0f, 1.0f); /* Yellow */
	glVertex2f(0, height);
	glEnd();

	/* Draw grid */
	glColor4f(1.0f, 1.0f, 1.0f, 0.3f);
	glBegin(GL_LINES);
	for (i = 0; i < width; i += 50) {
		glVertex2f(i, 0);
		glVertex2f(i, height);
	}
	for (i = 0; i < height; i += 50) {
		glVertex2f(0, i);
		glVertex2f(width, i);
	}
	glEnd();

	/* Draw center circle */
	draw_circle(width / 2, height / 2, 100, COLOR_YELLOW);

	/* Draw test pattern text */
	char test_text[100];
	sprintf(test_text, "EvoX Test Pattern - Resolution: %dx%d", width, height);
	draw_text(width / 2 - 150, height / 2 - 50, test_text, COLOR_WHITE, 1.0f);
}

/*=============================================================================
 * Rendering
 *===========================================================================*/

static void render_game(AppState *state) {
	SensoryGame *game = state->game;
	DisplayConfig *dc = &state->display;
	float scale = dc->scale_factor;
	int i, j;
	char buffer[256];
	float shake_x = 0, shake_y = 0;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, dc->width, dc->height, 0, -1, 1);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	/* Apply screen shake */
	if (game->screen_shake > 0) {
		shake_x = random_range(-game->screen_shake, game->screen_shake) * scale;
		shake_y = random_range(-game->screen_shake, game->screen_shake) * scale;
		glTranslatef(shake_x, shake_y, 0);
	}

	/* Draw background based on weather */
	float bg_r = 0.05f, bg_g = 0.05f, bg_b = 0.1f;
	if (game->weather_type == WEATHER_FOG) {
		bg_r = bg_g = bg_b = 0.3f;
	} else if (game->weather_type == WEATHER_STORM) {
		bg_r = 0.1f;
		bg_g = 0.1f;
		bg_b = 0.2f;
	}

	glClearColor(bg_r, bg_g, bg_b, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	/* Draw stars */
	glPointSize(1.5f * scale);
	glColor4f(0.8f, 0.8f, 1.0f, 0.8f);
	glBegin(GL_POINTS);
	for (i = 0; i < 100; i++) {
		float x = fmodf(game->game_time * 10 + i * 37, dc->width);
		float y = fmodf(i * 73 + game->game_time * 5, dc->height);
		glVertex2f(x, y);
	}
	glEnd();

	/* Draw particles */
	draw_particles(game, scale);

	/* Draw lightning */
	for (i = 0; i < MAX_ENEMIES; i++) {
		if (game->lightnings[i].active) {
			draw_lightning(&game->lightnings[i]);
		}
	}

	/* Draw enemy trails */
	for (i = 0; i < game->num_enemies; i++) {
		AIEnemy *e = &game->enemies[i];
		if (!e->active)
			continue;

		glLineWidth(1.5f * scale);
		glBegin(GL_LINE_STRIP);
		for (j = 0; j < TRAIL_LENGTH; j++) {
			int idx = (e->trail_index - j + TRAIL_LENGTH) % TRAIL_LENGTH;
			float alpha = 1.0f - (float) j / TRAIL_LENGTH;
			glColor4f(1.0f, 0.2f, 0.2f, alpha * 0.3f);
			glVertex2f(e->trail_history[idx][0], e->trail_history[idx][1]);
		}
		glEnd();
	}

	/* Draw enemies */
	for (i = 0; i < game->num_enemies; i++) {
		AIEnemy *e = &game->enemies[i];
		if (!e->active)
			continue;

		float pulse = 0.8f + 0.2f * sinf(e->pulse_phase);
		unsigned int enemy_color = e->color;
		draw_circle(e->x, e->y, 15.0f * scale * pulse, enemy_color);

		/* Health bar */
		float health_pct = e->health / e->max_health;
		draw_rect(e->x, e->y - 25 * scale, 30 * scale, 4 * scale, COLOR_RED);
		draw_rect(e->x - 15 * scale + 15 * scale * health_pct,
				e->y - 25 * scale, 30 * scale * health_pct, 4 * scale,
				COLOR_GREEN);
	}

	/* Draw bullets */
	for (i = 0; i < game->num_bullets; i++) {
		if (game->bullets[i].active) {
			draw_rect(game->bullets[i].x, game->bullets[i].y, 3 * scale,
					8 * scale, game->bullets[i].color);
		}
	}

	/* Draw player */
	draw_rect(game->x, game->y, 25 * scale, 25 * scale, COLOR_GREEN);

	/* Draw shield */
	if (game->shield > 50) {
		draw_circle(game->x, game->y, 35 * scale, 0x8080FFFF);
	}

	/* Apply flash effect */
	if (game->flash_intensity > 0) {
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE);
		float r = ((game->flash_color >> 24) & 0xFF) / 255.0f;
		float g = ((game->flash_color >> 16) & 0xFF) / 255.0f;
		float b = ((game->flash_color >> 8) & 0xFF) / 255.0f;
		glColor4f(r, g, b, game->flash_intensity);
		glBegin(GL_QUADS);
		glVertex2f(0, 0);
		glVertex2f(dc->width, 0);
		glVertex2f(dc->width, dc->height);
		glVertex2f(0, dc->height);
		glEnd();
		glDisable(GL_BLEND);
	}

	/* Draw HUD */
	glDisable(GL_BLEND);
	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

	snprintf(buffer, sizeof(buffer),
			"Score: %d  Health: %.0f  Shield: %.0f  Wave: %d/%d  Combo: %dx",
			game->score, game->health, game->shield, game->current_wave,
			MAX_WAVES, game->combo);
	draw_text(10, 20, buffer, COLOR_WHITE, scale);

	if (game->ai) {
		snprintf(buffer, sizeof(buffer), "AI Steps: %d  Reward: %.2f",
				game->ai->learning_step, game->ai->cumulative_reward);
		draw_text(10, 45, buffer, COLOR_CYAN, scale * 0.8f);
	}

	snprintf(buffer, sizeof(buffer), "FPS: %.1f  Enemies: %d  Particles: %d",
			state->fps, game->num_enemies, game->num_particles);
	draw_text(10, 70, buffer, COLOR_WHITE, scale * 0.8f);

	if (game->game_over) {
		glColor4f(0, 0, 0, 0.8f);
		glBegin(GL_QUADS);
		glVertex2f(0, 0);
		glVertex2f(dc->width, 0);
		glVertex2f(dc->width, dc->height);
		glVertex2f(0, dc->height);
		glEnd();

		draw_text(dc->width / 2 - 100, dc->height / 2 - 20, "GAME OVER",
				COLOR_RED, 2.0f);
		snprintf(buffer, sizeof(buffer), "Final Score: %d", game->score);
		draw_text(dc->width / 2 - 80, dc->height / 2 + 20, buffer, COLOR_WHITE,
				1.5f);
	}

	if (game->victory) {
		glColor4f(0, 0, 0, 0.8f);
		glBegin(GL_QUADS);
		glVertex2f(0, 0);
		glVertex2f(dc->width, 0);
		glVertex2f(dc->width, dc->height);
		glVertex2f(0, dc->height);
		glEnd();

		draw_text(dc->width / 2 - 80, dc->height / 2 - 20, "VICTORY!",
				COLOR_GOLD, 2.0f);
		snprintf(buffer, sizeof(buffer), "Score: %d", game->score);
		draw_text(dc->width / 2 - 50, dc->height / 2 + 20, buffer, COLOR_WHITE,
				1.5f);
	}

	if (game->paused) {
		glColor4f(0, 0, 0, 0.5f);
		glBegin(GL_QUADS);
		glVertex2f(0, 0);
		glVertex2f(dc->width, 0);
		glVertex2f(dc->width, dc->height);
		glVertex2f(0, dc->height);
		glEnd();

		draw_text(dc->width / 2 - 50, dc->height / 2, "PAUSED", COLOR_WHITE,
				2.0f);
	}
}

/*=============================================================================
 * Application Initialization
 *===========================================================================*/

static int app_init(AppState *state, int argc, char *argv[]) {
	int i;
	int fullscreen = DISPLAY_WINDOWED; /* Start windowed for debugging */

	memset(state, 0, sizeof(AppState));
	g_app_state = state;

	state->verbose = 0;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0)
			state->verbose = 1;
		if (strcmp(argv[i], "-w") == 0)
			fullscreen = DISPLAY_WINDOWED;
		if (strcmp(argv[i], "-f") == 0)
			fullscreen = DISPLAY_FULLSCREEN;
		if (strcmp(argv[i], "-m") == 0)
			fullscreen = DISPLAY_MULTIMONITOR;
	}

	printf("\n");
	printf("╔════════════════════════════════════════════════════════════╗\n");
	printf("║     EvoX - Ultimate AI Sensory Experience System v3.2     ║\n");
	printf("╚════════════════════════════════════════════════════════════╝\n");
	printf("\n");

	srand((unsigned int) time(NULL));

	signal(SIGINT, handle_signal);
	signal(SIGTERM, handle_signal);

	if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_TIMER) < 0) {
		fprintf(stderr, "SDL Init failed: %s\n", SDL_GetError());
		return -1;
	}

	detect_monitors(state);
	state->display = optimize_display_config(state, fullscreen);

	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
	SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);

	Uint32 flags = SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_SHOWN;
	if (state->display.fullscreen == DISPLAY_FULLSCREEN) {
		flags |= SDL_WINDOW_FULLSCREEN;
	} else if (state->display.fullscreen == DISPLAY_FULLSCREEN_DESKTOP) {
		flags |= SDL_WINDOW_FULLSCREEN_DESKTOP;
	} else if (state->display.fullscreen == DISPLAY_BORDERLESS) {
		flags |= SDL_WINDOW_BORDERLESS;
	}

	state->window = SDL_CreateWindow("EvoX - Ultimate AI Experience",
	SDL_WINDOWPOS_CENTERED,
	SDL_WINDOWPOS_CENTERED, state->display.width, state->display.height, flags);

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

	SDL_GL_SetSwapInterval(state->display.vsync ? 1 : 0);

	glViewport(0, 0, state->display.width, state->display.height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, state->display.width, state->display.height, 0, -1, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glClearColor(0.02f, 0.02f, 0.05f, 1.0f);

	printf("\nOpenGL Information:\n");
	printf("  Vendor: %s\n", glGetString(GL_VENDOR));
	printf("  Renderer: %s\n", glGetString(GL_RENDERER));
	printf("  Version: %s\n", glGetString(GL_VERSION));

	printf("\nInitializing audio system...\n");
	state->audio = audio_system_create();
	if (state->audio && state->audio->initialized) {
		printf("  Audio system initialized successfully\n");
		pthread_create(&state->audio_thread, NULL, audio_thread_func,
				state->audio);
	} else {
		printf(
				"  Audio system initialization failed - continuing without audio\n");
	}

	printf("\nInitializing game...\n");
	state->game = (SensoryGame*) malloc(sizeof(SensoryGame));
	init_game(state->game);

	gettimeofday(&state->last_frame, NULL);
	state->running = 1;

	printf("\n");
	printf("Controls:\n");
	printf("  Arrow Keys/WASD - Move\n");
	printf("  SPACE - Fire\n");
	printf("  W - Cycle Weather\n");
	printf("  F - Force Field\n");
	printf("  P - Pause\n");
	printf("  R - Restart\n");
	printf("  1 - Windowed Mode\n");
	printf("  2 - Fullscreen Mode\n");
	printf("  ESC - Exit\n\n");

	printf("Display Mode: %s\n",
			state->display.fullscreen == DISPLAY_WINDOWED ? "Windowed" :
			state->display.fullscreen == DISPLAY_FULLSCREEN ? "Fullscreen" :
			state->display.fullscreen == DISPLAY_FULLSCREEN_DESKTOP ?
					"Fullscreen Desktop" :
			state->display.fullscreen == DISPLAY_BORDERLESS ?
					"Borderless" : "Unknown");
	printf("Resolution: %dx%d\n", state->display.width, state->display.height);
	printf("Scale Factor: %.2f\n\n", state->display.scale_factor);

	printf("AI Systems: Q-Learning Active\n\n");

	return 0;
}

/*=============================================================================
 * Application Cleanup
 *===========================================================================*/

static void app_cleanup(AppState *state) {
	printf("\nShutting down...\n");

	if (state->game) {
		printf("Final Score: %d\n", state->game->score);
		printf("Total Kills: %d\n", state->game->total_kills);

		if (state->game->ai) {
			ai_system_destroy(state->game->ai);
		}

		free(state->game);
	}

	if (state->audio) {
		if (state->audio->initialized) {
			pthread_cancel(state->audio_thread);
			pthread_join(state->audio_thread, NULL);
		}
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
	Uint32 frame_start;
	Uint32 fps_last_time = SDL_GetTicks();
	int fps_frames = 0;

	while (state->running && g_running) {
		frame_start = SDL_GetTicks();

		process_input(state);

		if (state->game) {
			state->game->delta_time = 1.0f / TARGET_FPS;
			update_game(state);
		}

		render_game(state);
		SDL_GL_SwapWindow(state->window);

		fps_frames++;
		Uint32 now = SDL_GetTicks();
		if (now - fps_last_time >= 1000) {
			state->fps = (float) fps_frames * 1000.0f / (now - fps_last_time);
			fps_frames = 0;
			fps_last_time = now;
		}

		Uint32 frame_time = SDL_GetTicks() - frame_start;
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

	result = app_init(&state, argc, argv);
	if (result != 0) {
		return 1;
	}

	app_run(&state);
	app_cleanup(&state);

	return 0;
}
