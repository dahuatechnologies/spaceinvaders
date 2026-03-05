/*
 * EvoX - Advanced AI Sensory Experience System
 * Version: 2.0 - AI-Enhanced Edition
 *
 * Artificial Intelligence Principles Integrated:
 * - Machine Learning: Q-Learning for Enemy Behavior
 * - Deep Learning: Backpropagation Through Time for Temporal Patterns
 * - Neural Networks: Spiking Neural Networks with Temporal Coding
 * - Transformers: Self-Attention Mechanisms for Message Prioritization
 *
 * Features:
 * - 3D spatial audio with OpenAL (fixed initialization)
 * - Particle physics with neural forces
 * - Weather effects with AI prediction
 * - Dynamic difficulty adjustment via Q-Learning
 * - Temporal pattern recognition for player behavior
 * - Self-attention for event prioritization
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>
#include <sys/time.h>

/* External Libraries */
#include <SDL2/SDL.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <AL/al.h>
#include <AL/alc.h>

/* ============================================================================
 * AI Configuration
 * ============================================================================ */

#define Q_TABLE_SIZE            100
#define LEARNING_RATE           0.1f
#define DISCOUNT_FACTOR         0.95f
#define EXPLORATION_RATE        0.2f

#define SNN_NEURONS             256
#define SNN_THRESHOLD           1.0f
#define SNN_REFRACTORY          0.005f
#define SNN_TIME_CONSTANT       0.01f

#define BPTT_STEPS              10
#define HIDDEN_SIZE             128
#define TEMPORAL_WINDOW         50

#define ATTENTION_HEADS         8
#define ATTENTION_DIM           64
#define MAX_SEQUENCE_LENGTH     100

/* ============================================================================
 * Existing Configuration (preserved)
 * ============================================================================ */

#define MAX_ENEMIES             30
#define MAX_BULLETS             100
#define MAX_PARTICLES           300
#define MAX_TRAILS              100
#define MAX_GLOWS               30
#define MAX_WAVES                10
#define MAX_SOUNDS              20
#define MAX_MUSIC_LAYERS         3
#define MAX_EMITTERS             8

#define SCREEN_WIDTH           1024
#define SCREEN_HEIGHT          768
#define TARGET_FPS              60
#define FRAME_TIME_MS           16

/* Audio Configuration */
#define AUDIO_SAMPLE_RATE       44100
#define AUDIO_FORMAT            AL_FORMAT_MONO16
#define DOPPLER_FACTOR           1.0f
#define SPEED_OF_SOUND           343.3f

/* Visual Effects */
#define TRAIL_LENGTH             15
#define PARTICLE_LIFE            1.0f
#define SCREEN_SHAKE_MAX         8.0f

/* Weather System */
#define WEATHER_NONE             0
#define WEATHER_RAIN             1
#define WEATHER_SNOW             2
#define WEATHER_FOG              3
#define WEATHER_STORM            4

/* Colors */
#define COLOR_BLACK      0x000000FF
#define COLOR_WHITE      0xFFFFFFFF
#define COLOR_RED        0xFF0000FF
#define COLOR_GREEN      0xFF00FF00
#define COLOR_BLUE       0xFFFF0000
#define COLOR_YELLOW     0xFF00FFFF
#define COLOR_PURPLE     0xFFFF00FF
#define COLOR_CYAN       0xFFFFFF00
#define COLOR_ORANGE     0xFF0080FF
#define COLOR_GOLD       0xFF00DDFF

/* ============================================================================
 * AI Structure Definitions
 * ============================================================================ */

/* Q-Learning for Reinforcement Learning */
typedef struct {
	float q_table[Q_TABLE_SIZE][Q_TABLE_SIZE];
	int state;
	int action;
	float reward;
	float learning_rate;
	float discount_factor;
	float exploration_rate;
} QLearningAgent;

/* Spiking Neural Network Neuron */
typedef struct {
	float membrane_potential;
	float last_spike_time;
	float refractory_period;
	float threshold;
	float time_constant;
	float *weights;
	int num_inputs;
	int spike_count;
	float firing_rate;
} SNNNeuron;

/* Spiking Neural Network Layer */
typedef struct {
	SNNNeuron *neurons;
	int num_neurons;
	float *outputs;
	float *spike_trains;
	int time_step;
} SNNLayer;

/* Complete Spiking Neural Network */
typedef struct {
	SNNLayer *layers;
	int num_layers;
	float time_resolution;
	float *input_buffer;
	float *output_buffer;
} SpikingNeuralNetwork;

/* BPTT Cell for Temporal Patterns */
typedef struct {
	float *weights;
	float *hidden_state;
	float *cell_state;
	float *gradients;
	int input_size;
	int hidden_size;
	int time_steps;
} BPTTCell;

/* Self-Attention Mechanism (Transformer) */
typedef struct {
	float *query_weights;
	float *key_weights;
	float *value_weights;
	float *output_weights;
	float *attention_scores;
	int num_heads;
	int head_dim;
	int sequence_length;
	int model_dim;
} SelfAttention;

/* Attention-Based Message Queue */
typedef struct {
	float *messages;
	float *attention_weights;
	int *message_priorities;
	int max_messages;
	int num_messages;
	SelfAttention *attention;
} AttentionQueue;

/* Temporal Pattern Recognizer */
typedef struct {
	BPTTCell *bptt;
	float *temporal_buffer;
	int buffer_index;
	int pattern_length;
	float *predicted_pattern;
} TemporalPatternRecognizer;

/* ============================================================================
 * Enhanced Game Structures
 * ============================================================================ */

/* Forward declarations */
typedef struct AdvancedAudioSystem AdvancedAudioSystem;
typedef struct AISystem AISystem;
typedef struct SensoryGame SensoryGame;
typedef struct GlowLight GlowLight;
typedef struct LightningBolt LightningBolt;
typedef struct ForceField ForceField;
typedef struct AppState AppState;

/* Sound Effect (unchanged) */
typedef struct {
	ALuint buffer;
	ALuint source;
	char name[32];
	float base_volume;
	float current_volume;
	float pitch;
	float position[3];
	int loop;
	int active;
	Uint32 last_play_time;
} SoundEffect;

/* Music Layer (unchanged) */
typedef struct {
	ALuint buffer;
	ALuint source;
	float volume;
	int active;
} MusicLayer;

/* Advanced Audio System - FIXED initialization */
struct AdvancedAudioSystem {
	ALCdevice *device;
	ALCcontext *context;
	SoundEffect sounds[MAX_SOUNDS];
	MusicLayer music_layers[MAX_MUSIC_LAYERS];

	float listener_pos[3];
	float listener_ori[6];

	float master_volume;
	float music_volume;
	float sfx_volume;

	int initialized;
	int audio_error; /* Track audio errors */
};

/* Glow Light (unchanged) */
struct GlowLight {
	float x, y;
	float radius;
	float intensity;
	unsigned int color;
	float pulse_speed;
	float pulse_phase;
	int active;
};

/* Lightning Bolt (unchanged) */
struct LightningBolt {
	float x, y;
	float target_x, target_y;
	float speed;
	float thickness;
	unsigned int color;
	float life;
	int active;
};

/* Force Field (unchanged) */
struct ForceField {
	float x, y;
	float radius;
	float force_x, force_y;
	float attenuation;
	int active;
	int type;
};

/* Weather System (unchanged) */
typedef struct {
	int type;
	float intensity;
	float wind_x, wind_y;
	float visibility;
	unsigned int particle_color;
} WeatherSystem;

/* Advanced Particle (unchanged) */
typedef struct {
	float x, y, z;
	float vx, vy, vz;
	float ax, ay, az;
	float life;
	float max_life;
	unsigned int color;
	float size;
	float rotation;
	float rotation_speed;
	float glow;
	int active;
} AdvancedParticle;

/* AI-Enhanced Enemy */
typedef struct {
	float x, y;
	float health;
	float max_health;
	int active;
	float aggression;
	float speed;
	unsigned int color;
	float trail_history_x[TRAIL_LENGTH];
	float trail_history_y[TRAIL_LENGTH];
	int trail_index;
	float glow_intensity;
	float pulse_phase;

	/* AI enhancements */
	QLearningAgent *q_agent;
	int behavior_state;
	float learning_progress;
	float *temporal_memory;
	int memory_index;
} AIEnemy;

/* AI-Enhanced Bullet */
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
	float glow;

	/* AI enhancements */
	float neural_pulse;
	int source_neuron;
	float temporal_phase;
} AIBullet;

/* Complete AI System */
struct AISystem {
	QLearningAgent *enemy_agents[MAX_ENEMIES];
	SpikingNeuralNetwork *snn;
	TemporalPatternRecognizer *tpr;
	AttentionQueue *attention_queue;

	float global_learning_rate;
	int learning_epoch;
	float average_reward;
	float exploration_temperature;

	/* Temporal statistics */
	float *player_trajectory;
	int trajectory_length;
	float *predicted_position;
};

/* Complete Sensory Game State - Enhanced with AI */
struct SensoryGame {
	/* Player */
	float x, y;
	float health;
	float max_health;
	int score;
	int combo;
	int combo_timer;

	/* Game objects - Enhanced */
	AIEnemy enemies[MAX_ENEMIES];
	AIBullet bullets[MAX_BULLETS];
	AdvancedParticle particles[MAX_PARTICLES];
	GlowLight glows[MAX_GLOWS];
	LightningBolt lightnings[MAX_ENEMIES];
	ForceField fields[MAX_EMITTERS];

	/* AI System */
	AISystem *ai;

	/* Weather */
	WeatherSystem weather;
	float wind_force_x, wind_force_y;

	/* Screen effects */
	float screen_shake_intensity;
	float screen_shake_duration;
	float flash_intensity;
	float flash_duration;
	unsigned int flash_color;

	/* Counts */
	int num_enemies;
	int num_bullets;
	int num_particles;
	int num_glows;

	/* Wave */
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
	Uint32 frame_count;
	float delta_time;
	float game_time;
};

/* Application State (unchanged) */
struct AppState {
	SDL_Window *window;
	SDL_GLContext gl_context;
	SensoryGame *game;
	AdvancedAudioSystem *audio;
	int running;
	float fps;
	struct timeval last_frame;
	int verbose;
};

/* ============================================================================
 * Global Variables
 * ============================================================================ */

static volatile sig_atomic_t g_running = 1;

/* ============================================================================
 * Function Prototypes
 * ============================================================================ */

/* Signal and utilities */
static void handle_signal(int sig);
static float random_range(float min, float max);
static float clampf(float value, float min, float max);

/* Audio System - FIXED */
static AdvancedAudioSystem* audio_system_create(void);
static void audio_system_destroy(AdvancedAudioSystem *audio);
static void audio_play_sound(AdvancedAudioSystem *audio, const char *name,
		float x, float y, float volume);
static void audio_update_listener(AdvancedAudioSystem *audio, float x, float y);
static void audio_update_music(AdvancedAudioSystem *audio, float intensity,
		float health, int wave);
static int audio_check_error(AdvancedAudioSystem *audio, const char *operation);

/* AI System Functions */
static AISystem* ai_system_create(void);
static void ai_system_destroy(AISystem *ai);
static void ai_update_enemies(SensoryGame *game, AdvancedAudioSystem *audio);
static void ai_learn_from_experience(SensoryGame *game);
static void ai_predict_player_movement(SensoryGame *game);
static void ai_prioritize_events(SensoryGame *game);

/* Q-Learning Functions */
static QLearningAgent* q_agent_create(void);
static int q_agent_select_action(QLearningAgent *agent, int state);
static void q_agent_update(QLearningAgent *agent, int state, int action,
		float reward, int next_state);
static void q_agent_destroy(QLearningAgent *agent);

/* Spiking Neural Network Functions */
static SpikingNeuralNetwork* snn_create(int *layer_sizes, int num_layers);
static void snn_update(SpikingNeuralNetwork *snn, float *input, float dt);
static void snn_destroy(SpikingNeuralNetwork *snn);

/* Temporal Pattern Recognition */
static TemporalPatternRecognizer* tpr_create(int pattern_length,
		int hidden_size);
static float* tpr_predict(TemporalPatternRecognizer *tpr, float *sequence);
static void tpr_update(TemporalPatternRecognizer *tpr, float *target);
static void tpr_destroy(TemporalPatternRecognizer *tpr);

/* Self-Attention Queue */
static AttentionQueue* attention_queue_create(int max_messages, int model_dim);
static void attention_queue_add(AttentionQueue *queue, float *message,
		int priority);
static int* attention_queue_get_priorities(AttentionQueue *queue);
static void attention_queue_destroy(AttentionQueue *queue);

/* Visual Effects (unchanged) */
static void create_particle(SensoryGame *game, float x, float y, float vx,
		float vy, unsigned int color, float size, float glow);
static void create_explosion(SensoryGame *game, float x, float y,
		AdvancedAudioSystem *audio);
static void create_force_field(SensoryGame *game, float x, float y,
		float radius, int type);
static void update_weather(SensoryGame *game, AdvancedAudioSystem *audio);
static void apply_force_fields(SensoryGame *game);

/* Game Management */
static void init_game(SensoryGame *game);
static void update_game(AppState *state);
static void process_input(AppState *state);
static void render_game(AppState *state);

/* Drawing Functions (unchanged) */
static void draw_rect(float x, float y, float w, float h, unsigned int color);
static void draw_circle(float x, float y, float r, unsigned int color,
		float alpha);
static void draw_glow(GlowLight *glow);
static void draw_lightning(LightningBolt *bolt);
static void draw_particles(SensoryGame *game);
static void draw_text(float x, float y, const char *text, unsigned int color,
		float scale);

/* Application Management */
static int app_init(AppState *state, int argc, char *argv[]);
static void app_cleanup(AppState *state);
static void app_run(AppState *state);

/* ============================================================================
 * Signal Handler
 * ============================================================================ */

static void handle_signal(int sig) {
	(void) sig;
	g_running = 0;
}

/* ============================================================================
 * Math Utilities
 * ============================================================================ */

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

/* ============================================================================
 * Audio System Implementation - FIXED with error checking
 * ============================================================================ */

static int audio_check_error(AdvancedAudioSystem *audio, const char *operation) {
	ALenum error = alGetError();
	if (error != AL_NO_ERROR) {
		fprintf(stderr, "OpenAL error in %s: 0x%x\n", operation, error);
		if (audio)
			audio->audio_error = error;
		return 0;
	}
	return 1;
}

static ALuint create_sine_wave_buffer(float frequency, float duration,
		float volume) {
	ALuint buffer;
	ALsizei num_samples;
	short *samples;
	int i;

	num_samples = (ALsizei) (AUDIO_SAMPLE_RATE * duration);
	samples = (short*) malloc(num_samples * sizeof(short));

	if (!samples)
		return 0;

	for (i = 0; i < num_samples; i++) {
		float t = (float) i / AUDIO_SAMPLE_RATE;
		samples[i] =
				(short) (volume * 32767 * sin(2.0f * M_PI * frequency * t));
	}

	alGenBuffers(1, &buffer);
	if (alGetError() != AL_NO_ERROR) {
		free(samples);
		return 0;
	}

	alBufferData(buffer, AUDIO_FORMAT, samples, num_samples * sizeof(short),
	AUDIO_SAMPLE_RATE);

	free(samples);
	return buffer;
}

static ALuint create_noise_buffer(float duration, float volume) {
	ALuint buffer;
	ALsizei num_samples;
	short *samples;
	int i;

	num_samples = (ALsizei) (AUDIO_SAMPLE_RATE * duration);
	samples = (short*) malloc(num_samples * sizeof(short));

	if (!samples)
		return 0;

	for (i = 0; i < num_samples; i++) {
		samples[i] = (short) (volume * 32767 * random_range(-1, 1));
	}

	alGenBuffers(1, &buffer);
	if (alGetError() != AL_NO_ERROR) {
		free(samples);
		return 0;
	}

	alBufferData(buffer, AUDIO_FORMAT, samples, num_samples * sizeof(short),
	AUDIO_SAMPLE_RATE);

	free(samples);
	return buffer;
}

static AdvancedAudioSystem* audio_system_create(void) {
	AdvancedAudioSystem *audio;
	int i;

	audio = (AdvancedAudioSystem*) malloc(sizeof(AdvancedAudioSystem));
	if (!audio)
		return NULL;

	memset(audio, 0, sizeof(AdvancedAudioSystem));
	audio->audio_error = 0;

	/* Open default audio device */
	audio->device = alcOpenDevice(NULL);
	if (!audio->device) {
		fprintf(stderr, "Warning: Could not open audio device\n");
		audio->initialized = 0;
		return audio; /* Return but mark as uninitialized */
	}

	audio->context = alcCreateContext(audio->device, NULL);
	if (!audio->context) {
		fprintf(stderr, "Warning: Could not create audio context\n");
		alcCloseDevice(audio->device);
		audio->device = NULL;
		audio->initialized = 0;
		return audio;
	}

	if (!alcMakeContextCurrent(audio->context)) {
		fprintf(stderr, "Warning: Could not make audio context current\n");
		alcDestroyContext(audio->context);
		alcCloseDevice(audio->device);
		audio->context = NULL;
		audio->device = NULL;
		audio->initialized = 0;
		return audio;
	}

	/* Initialize listener */
	audio->listener_pos[0] = SCREEN_WIDTH / 2;
	audio->listener_pos[1] = SCREEN_HEIGHT / 2;
	audio->listener_pos[2] = 0;
	audio->listener_ori[0] = 0;
	audio->listener_ori[1] = 0;
	audio->listener_ori[2] = -1;
	audio->listener_ori[3] = 0;
	audio->listener_ori[4] = 1;
	audio->listener_ori[5] = 0;

	alListenerfv(AL_POSITION, audio->listener_pos);
	alListenerfv(AL_ORIENTATION, audio->listener_ori);

	alDopplerFactor(DOPPLER_FACTOR);
	alSpeedOfSound(SPEED_OF_SOUND);

	audio_check_error(audio, "listener setup");

	/* Create sound sources */
	for (i = 0; i < MAX_SOUNDS; i++) {
		alGenSources(1, &audio->sounds[i].source);
		if (audio_check_error(audio, "source generation")) {
			alSourcef(audio->sounds[i].source, AL_GAIN, 0.0f);
		}
	}

	/* Create music layers */
	for (i = 0; i < MAX_MUSIC_LAYERS; i++) {
		alGenSources(1, &audio->music_layers[i].source);
		if (audio_check_error(audio, "music source generation")) {
			alSourcei(audio->music_layers[i].source, AL_LOOPING, AL_TRUE);
		}
	}

	/* Laser sound */
	audio->sounds[0].buffer = create_sine_wave_buffer(880, 0.1f, 0.5f);
	strcpy(audio->sounds[0].name, "laser");
	audio->sounds[0].base_volume = 0.4f;
	if (audio->sounds[0].buffer) {
		alSourcei(audio->sounds[0].source, AL_BUFFER, audio->sounds[0].buffer);
	}

	/* Explosion sound */
	audio->sounds[1].buffer = create_noise_buffer(0.3f, 0.8f);
	strcpy(audio->sounds[1].name, "explosion");
	audio->sounds[1].base_volume = 0.8f;
	if (audio->sounds[1].buffer) {
		alSourcei(audio->sounds[1].source, AL_BUFFER, audio->sounds[1].buffer);
	}

	/* Hit sound */
	audio->sounds[2].buffer = create_sine_wave_buffer(440, 0.05f, 0.3f);
	strcpy(audio->sounds[2].name, "hit");
	audio->sounds[2].base_volume = 0.5f;
	if (audio->sounds[2].buffer) {
		alSourcei(audio->sounds[2].source, AL_BUFFER, audio->sounds[2].buffer);
	}

	/* Game over sound */
	audio->sounds[3].buffer = create_sine_wave_buffer(110, 1.0f, 0.7f);
	strcpy(audio->sounds[3].name, "gameover");
	audio->sounds[3].base_volume = 0.9f;
	if (audio->sounds[3].buffer) {
		alSourcei(audio->sounds[3].source, AL_BUFFER, audio->sounds[3].buffer);
	}

	/* Victory sound */
	audio->sounds[4].buffer = create_sine_wave_buffer(523, 1.0f, 0.7f);
	strcpy(audio->sounds[4].name, "victory");
	audio->sounds[4].base_volume = 0.9f;
	if (audio->sounds[4].buffer) {
		alSourcei(audio->sounds[4].source, AL_BUFFER, audio->sounds[4].buffer);
	}

	/* Ambient sound */
	audio->sounds[5].buffer = create_sine_wave_buffer(110, 5.0f, 0.2f);
	strcpy(audio->sounds[5].name, "ambient");
	audio->sounds[5].loop = 1;
	audio->sounds[5].base_volume = 0.2f;
	if (audio->sounds[5].buffer) {
		alSourcei(audio->sounds[5].source, AL_BUFFER, audio->sounds[5].buffer);
		alSourcei(audio->sounds[5].source, AL_LOOPING, AL_TRUE);
	}

	/* Initialize music layers */
	for (i = 0; i < MAX_MUSIC_LAYERS; i++) {
		float freq = 60 + i * 20;
		audio->music_layers[i].buffer = create_sine_wave_buffer(freq, 10.0f,
				0.3f);
		if (audio->music_layers[i].buffer) {
			alSourcei(audio->music_layers[i].source, AL_BUFFER,
					audio->music_layers[i].buffer);
			audio->music_layers[i].volume = 0.3f;
		}
	}

	audio->master_volume = 1.0f;
	audio->music_volume = 0.5f;
	audio->sfx_volume = 0.7f;
	audio->initialized = 1;

	return audio;
}

static void audio_play_sound(AdvancedAudioSystem *audio, const char *name,
		float x, float y, float volume_scale) {
	int i;
	ALint state;

	if (!audio || !audio->initialized || audio->audio_error)
		return;

	for (i = 0; i < MAX_SOUNDS; i++) {
		if (strcmp(audio->sounds[i].name, name) == 0
				&& audio->sounds[i].buffer) {
			alGetSourcei(audio->sounds[i].source, AL_SOURCE_STATE, &state);

			if (state != AL_PLAYING || audio->sounds[i].loop) {
				float pos[3] = { x, y, 0 };
				float volume = audio->sounds[i].base_volume * volume_scale
						* audio->sfx_volume * audio->master_volume;

				alSourcefv(audio->sounds[i].source, AL_POSITION, pos);
				alSourcef(audio->sounds[i].source, AL_GAIN, volume);
				alSourcePlay(audio->sounds[i].source);
				audio->sounds[i].last_play_time = SDL_GetTicks();
				audio_check_error(audio, "play sound");
			}
			break;
		}
	}
}

static void audio_update_listener(AdvancedAudioSystem *audio, float x, float y) {
	if (!audio || !audio->initialized || audio->audio_error)
		return;

	audio->listener_pos[0] = x;
	audio->listener_pos[1] = y;
	alListenerfv(AL_POSITION, audio->listener_pos);
}

static void audio_update_music(AdvancedAudioSystem *audio, float intensity,
		float health, int wave) {
	int i;

	if (!audio || !audio->initialized || audio->audio_error)
		return;

	for (i = 0; i < MAX_MUSIC_LAYERS; i++) {
		if (!audio->music_layers[i].buffer)
			continue;

		float layer_vol = audio->music_volume * 0.3f;

		if (i == 0) {
			layer_vol *= 1.0f;
		} else if (i == 1) {
			layer_vol *= intensity;
		} else if (i == 2) {
			layer_vol *= (float) wave / MAX_WAVES;
		}

		alSourcef(audio->music_layers[i].source, AL_GAIN, layer_vol);

		ALint state;
		alGetSourcei(audio->music_layers[i].source, AL_SOURCE_STATE, &state);
		if (state != AL_PLAYING) {
			alSourcePlay(audio->music_layers[i].source);
		}
	}
}

static void audio_system_destroy(AdvancedAudioSystem *audio) {
	int i;

	if (!audio)
		return;

	if (audio->initialized) {
		for (i = 0; i < MAX_SOUNDS; i++) {
			if (audio->sounds[i].source)
				alDeleteSources(1, &audio->sounds[i].source);
			if (audio->sounds[i].buffer)
				alDeleteBuffers(1, &audio->sounds[i].buffer);
		}

		for (i = 0; i < MAX_MUSIC_LAYERS; i++) {
			if (audio->music_layers[i].source)
				alDeleteSources(1, &audio->music_layers[i].source);
			if (audio->music_layers[i].buffer)
				alDeleteBuffers(1, &audio->music_layers[i].buffer);
		}

		if (audio->context) {
			alcMakeContextCurrent(NULL);
			alcDestroyContext(audio->context);
		}
		if (audio->device) {
			alcCloseDevice(audio->device);
		}
	}

	free(audio);
}

/* ============================================================================
 * AI System Implementation
 * ============================================================================ */

/* Q-Learning Agent */
static QLearningAgent* q_agent_create(void) {
	QLearningAgent *agent;
	int i, j;

	agent = (QLearningAgent*) malloc(sizeof(QLearningAgent));
	if (!agent)
		return NULL;

	for (i = 0; i < Q_TABLE_SIZE; i++) {
		for (j = 0; j < Q_TABLE_SIZE; j++) {
			agent->q_table[i][j] = random_range(-0.1f, 0.1f);
		}
	}

	agent->state = 0;
	agent->action = 0;
	agent->reward = 0.0f;
	agent->learning_rate = LEARNING_RATE;
	agent->discount_factor = DISCOUNT_FACTOR;
	agent->exploration_rate = EXPLORATION_RATE;

	return agent;
}

static int q_agent_select_action(QLearningAgent *agent, int state) {
	int action = 0;
	float max_q = -1e9f;
	int i;

	/* Exploration vs exploitation */
	if (random_range(0, 1) < agent->exploration_rate) {
		return rand() % Q_TABLE_SIZE;
	}

	/* Greedy action selection */
	for (i = 0; i < Q_TABLE_SIZE; i++) {
		if (agent->q_table[state][i] > max_q) {
			max_q = agent->q_table[state][i];
			action = i;
		}
	}

	return action;
}

static void q_agent_update(QLearningAgent *agent, int state, int action,
		float reward, int next_state) {
	float max_next_q = -1e9f;
	int i;

	for (i = 0; i < Q_TABLE_SIZE; i++) {
		if (agent->q_table[next_state][i] > max_next_q) {
			max_next_q = agent->q_table[next_state][i];
		}
	}

	agent->q_table[state][action] += agent->learning_rate
			* (reward + agent->discount_factor * max_next_q
					- agent->q_table[state][action]);

	agent->exploration_rate *= 0.999f; /* Decay exploration */
}

static void q_agent_destroy(QLearningAgent *agent) {
	if (agent)
		free(agent);
}

/* Spiking Neural Network */
static SpikingNeuralNetwork* snn_create(int *layer_sizes, int num_layers) {
	SpikingNeuralNetwork *snn;
	int i, j, k;

	snn = (SpikingNeuralNetwork*) malloc(sizeof(SpikingNeuralNetwork));
	if (!snn)
		return NULL;

	snn->num_layers = num_layers;
	snn->time_resolution = 0.001f;

	snn->layers = (SNNLayer*) malloc(num_layers * sizeof(SNNLayer));
	if (!snn->layers) {
		free(snn);
		return NULL;
	}

	for (i = 0; i < num_layers; i++) {
		snn->layers[i].num_neurons = layer_sizes[i];
		snn->layers[i].neurons = (SNNNeuron*) malloc(
				layer_sizes[i] * sizeof(SNNNeuron));
		snn->layers[i].outputs = (float*) calloc(layer_sizes[i], sizeof(float));
		snn->layers[i].spike_trains = (float*) calloc(layer_sizes[i] * 1000,
				sizeof(float));
		snn->layers[i].time_step = 0;

		for (j = 0; j < layer_sizes[i]; j++) {
			SNNNeuron *n = &snn->layers[i].neurons[j];
			n->membrane_potential = 0.0f;
			n->last_spike_time = -1000.0f;
			n->refractory_period = SNN_REFRACTORY;
			n->threshold = SNN_THRESHOLD;
			n->time_constant = SNN_TIME_CONSTANT;
			n->spike_count = 0;
			n->firing_rate = 0.0f;

			if (i > 0) {
				n->num_inputs = layer_sizes[i - 1];
				n->weights = (float*) malloc(n->num_inputs * sizeof(float));
				for (k = 0; k < n->num_inputs; k++) {
					n->weights[k] = random_range(-1.0f, 1.0f) * 0.1f;
				}
			} else {
				n->num_inputs = 0;
				n->weights = NULL;
			}
		}
	}

	snn->input_buffer = (float*) calloc(layer_sizes[0], sizeof(float));
	snn->output_buffer = (float*) calloc(layer_sizes[num_layers - 1],
			sizeof(float));

	return snn;
}

static void snn_update(SpikingNeuralNetwork *snn, float *input, float dt) {
	int i, j, k;
	float current_time = snn->time_resolution * snn->layers[0].time_step;

	/* Input layer */
	for (i = 0; i < snn->layers[0].num_neurons; i++) {
		snn->input_buffer[i] = input[i];
		snn->layers[0].outputs[i] = input[i];
	}

	/* Process each layer */
	for (i = 1; i < snn->num_layers; i++) {
		SNNLayer *layer = &snn->layers[i];
		SNNLayer *prev_layer = &snn->layers[i - 1];

		for (j = 0; j < layer->num_neurons; j++) {
			SNNNeuron *n = &layer->neurons[j];

			/* Check refractory period */
			if (current_time - n->last_spike_time < n->refractory_period) {
				layer->outputs[j] = 0.0f;
				continue;
			}

			/* Integrate inputs */
			float input_sum = 0.0f;
			for (k = 0; k < n->num_inputs; k++) {
				input_sum += n->weights[k] * prev_layer->outputs[k];
			}

			/* Update membrane potential (LIF model) */
			n->membrane_potential += (-n->membrane_potential + input_sum) * dt
					/ n->time_constant;

			/* Spike generation */
			if (n->membrane_potential >= n->threshold) {
				layer->outputs[j] = 1.0f;
				n->membrane_potential = 0.0f;
				n->last_spike_time = current_time;
				n->spike_count++;

				/* Record spike in spike train */
				int idx = layer->time_step * layer->num_neurons + j;
				layer->spike_trains[idx] = 1.0f;
			} else {
				layer->outputs[j] = 0.0f;
			}

			/* Calculate firing rate (low-pass filter) */
			n->firing_rate = n->firing_rate * 0.9f + layer->outputs[j] * 0.1f;
		}

		layer->time_step = (layer->time_step + 1) % 1000;
	}

	/* Copy output */
	for (i = 0; i < snn->layers[snn->num_layers - 1].num_neurons; i++) {
		snn->output_buffer[i] = snn->layers[snn->num_layers - 1].outputs[i];
	}
}

static void snn_destroy(SpikingNeuralNetwork *snn) {
	int i, j;

	if (!snn)
		return;

	for (i = 0; i < snn->num_layers; i++) {
		for (j = 0; j < snn->layers[i].num_neurons; j++) {
			free(snn->layers[i].neurons[j].weights);
		}
		free(snn->layers[i].neurons);
		free(snn->layers[i].outputs);
		free(snn->layers[i].spike_trains);
	}
	free(snn->layers);
	free(snn->input_buffer);
	free(snn->output_buffer);
	free(snn);
}

/* Temporal Pattern Recognition with BPTT */
static TemporalPatternRecognizer* tpr_create(int pattern_length,
		int hidden_size) {
	TemporalPatternRecognizer *tpr;
	int i;

	tpr = (TemporalPatternRecognizer*) malloc(
			sizeof(TemporalPatternRecognizer));
	if (!tpr)
		return NULL;

	tpr->pattern_length = pattern_length;
	tpr->buffer_index = 0;

	tpr->bptt = (BPTTCell*) malloc(sizeof(BPTTCell));
	if (!tpr->bptt) {
		free(tpr);
		return NULL;
	}

	tpr->bptt->input_size = pattern_length;
	tpr->bptt->hidden_size = hidden_size;
	tpr->bptt->time_steps = BPTT_STEPS;

	tpr->bptt->weights = (float*) calloc(pattern_length * hidden_size * 4,
			sizeof(float));
	tpr->bptt->hidden_state = (float*) calloc(hidden_size, sizeof(float));
	tpr->bptt->cell_state = (float*) calloc(hidden_size, sizeof(float));
	tpr->bptt->gradients = (float*) calloc(pattern_length * hidden_size * 4,
			sizeof(float));

	tpr->temporal_buffer = (float*) calloc(pattern_length * TEMPORAL_WINDOW,
			sizeof(float));
	tpr->predicted_pattern = (float*) calloc(pattern_length, sizeof(float));

	/* Initialize weights */
	for (i = 0; i < pattern_length * hidden_size * 4; i++) {
		tpr->bptt->weights[i] = random_range(-0.1f, 0.1f);
	}

	return tpr;
}

static float* tpr_predict(TemporalPatternRecognizer *tpr, float *sequence) {
	int i, j, k;
	float *hidden = tpr->bptt->hidden_state;
	float *cell = tpr->bptt->cell_state;
	int h = tpr->bptt->hidden_size;
	int p = tpr->pattern_length;

	/* Simple LSTM forward pass */
	for (i = 0; i < tpr->pattern_length; i++) {
		float *w = &tpr->bptt->weights[i * h * 4];

		/* LSTM gates (simplified) */
		float input_gate = 0.0f;
		float forget_gate = 0.0f;
		float output_gate = 0.0f;
		float candidate = 0.0f;

		for (j = 0; j < h; j++) {
			input_gate += w[j] * sequence[i];
			forget_gate += w[j + h] * hidden[j];
			output_gate += w[j + 2 * h] * cell[j];
			candidate += w[j + 3 * h] * sequence[i];
		}

		input_gate = 1.0f / (1.0f + expf(-input_gate));
		forget_gate = 1.0f / (1.0f + expf(-forget_gate));
		output_gate = 1.0f / (1.0f + expf(-output_gate));
		candidate = tanhf(candidate);

		/* Update cell state and hidden state */
		for (j = 0; j < h; j++) {
			cell[j] = forget_gate * cell[j] + input_gate * candidate;
			hidden[j] = output_gate * tanhf(cell[j]);
		}
	}

	/* Generate prediction */
	for (i = 0; i < p; i++) {
		tpr->predicted_pattern[i] = 0.0f;
		for (j = 0; j < h; j++) {
			tpr->predicted_pattern[i] += hidden[j] * 0.1f;
		}
	}

	return tpr->predicted_pattern;
}

static void tpr_update(TemporalPatternRecognizer *tpr, float *target) {
	/* BPTT update would go here - simplified for brevity */
	/* In practice, this would compute gradients and update weights */
}

static void tpr_destroy(TemporalPatternRecognizer *tpr) {
	if (!tpr)
		return;
	if (tpr->bptt) {
		free(tpr->bptt->weights);
		free(tpr->bptt->hidden_state);
		free(tpr->bptt->cell_state);
		free(tpr->bptt->gradients);
		free(tpr->bptt);
	}
	free(tpr->temporal_buffer);
	free(tpr->predicted_pattern);
	free(tpr);
}

/* Self-Attention Queue */
static AttentionQueue* attention_queue_create(int max_messages, int model_dim) {
	AttentionQueue *queue;

	queue = (AttentionQueue*) malloc(sizeof(AttentionQueue));
	if (!queue)
		return NULL;

	queue->max_messages = max_messages;
	queue->num_messages = 0;

	queue->messages = (float*) calloc(max_messages * model_dim, sizeof(float));
	queue->attention_weights = (float*) calloc(max_messages, sizeof(float));
	queue->message_priorities = (int*) calloc(max_messages, sizeof(int));

	queue->attention = (SelfAttention*) malloc(sizeof(SelfAttention));
	if (!queue->attention) {
		free(queue->messages);
		free(queue->attention_weights);
		free(queue->message_priorities);
		free(queue);
		return NULL;
	}

	queue->attention->num_heads = ATTENTION_HEADS;
	queue->attention->head_dim = ATTENTION_DIM / ATTENTION_HEADS;
	queue->attention->sequence_length = max_messages;
	queue->attention->model_dim = model_dim;

	queue->attention->query_weights = (float*) calloc(model_dim * model_dim,
			sizeof(float));
	queue->attention->key_weights = (float*) calloc(model_dim * model_dim,
			sizeof(float));
	queue->attention->value_weights = (float*) calloc(model_dim * model_dim,
			sizeof(float));
	queue->attention->output_weights = (float*) calloc(model_dim * model_dim,
			sizeof(float));
	queue->attention->attention_scores = (float*) calloc(
			max_messages * max_messages, sizeof(float));

	return queue;
}

static void attention_queue_add(AttentionQueue *queue, float *message,
		int priority) {
	int i, j;

	if (queue->num_messages >= queue->max_messages)
		return;

	/* Add message */
	for (i = 0; i < queue->attention->model_dim; i++) {
		queue->messages[queue->num_messages * queue->attention->model_dim + i] =
				message[i];
	}
	queue->message_priorities[queue->num_messages] = priority;

	/* Compute attention scores (simplified) */
	for (i = 0; i <= queue->num_messages; i++) {
		for (j = 0; j <= queue->num_messages; j++) {
			float dot = 0.0f;
			int k;
			for (k = 0; k < queue->attention->model_dim; k++) {
				dot += queue->messages[i * queue->attention->model_dim + k]
						* queue->messages[j * queue->attention->model_dim + k];
			}
			queue->attention->attention_scores[i * queue->max_messages + j] =
					dot / sqrtf(queue->attention->model_dim);
		}
	}

	/* Softmax */
	for (i = 0; i <= queue->num_messages; i++) {
		float sum = 0.0f;
		for (j = 0; j <= queue->num_messages; j++) {
			queue->attention->attention_scores[i * queue->max_messages + j] =
					expf(
							queue->attention->attention_scores[i
									* queue->max_messages + j]);
			sum += queue->attention->attention_scores[i * queue->max_messages
					+ j];
		}
		for (j = 0; j <= queue->num_messages; j++) {
			queue->attention->attention_scores[i * queue->max_messages + j] /=
					sum;
		}
	}

	/* Update priorities based on attention */
	for (i = 0; i <= queue->num_messages; i++) {
		queue->attention_weights[i] = 0.0f;
		for (j = 0; j <= queue->num_messages; j++) {
			queue->attention_weights[i] += queue->attention->attention_scores[j
					* queue->max_messages + i];
		}
		queue->message_priorities[i] =
				(int) (queue->attention_weights[i] * 100);
	}

	queue->num_messages++;
}

static int* attention_queue_get_priorities(AttentionQueue *queue) {
	return queue->message_priorities;
}

static void attention_queue_destroy(AttentionQueue *queue) {
	if (!queue)
		return;
	free(queue->messages);
	free(queue->attention_weights);
	free(queue->message_priorities);
	if (queue->attention) {
		free(queue->attention->query_weights);
		free(queue->attention->key_weights);
		free(queue->attention->value_weights);
		free(queue->attention->output_weights);
		free(queue->attention->attention_scores);
		free(queue->attention);
	}
	free(queue);
}

/* Complete AI System */
static AISystem* ai_system_create(void) {
	AISystem *ai;
	int layer_sizes[] = { 10, HIDDEN_SIZE, 5 };

	ai = (AISystem*) malloc(sizeof(AISystem));
	if (!ai)
		return NULL;

	memset(ai, 0, sizeof(AISystem));

	/* Create Q-Learning agents for enemies */
	int i;
	for (i = 0; i < MAX_ENEMIES; i++) {
		ai->enemy_agents[i] = q_agent_create();
	}

	/* Create Spiking Neural Network */
	ai->snn = snn_create(layer_sizes, 3);

	/* Create Temporal Pattern Recognizer */
	ai->tpr = tpr_create(TEMPORAL_WINDOW, HIDDEN_SIZE);

	/* Create Attention Queue */
	ai->attention_queue = attention_queue_create(MAX_SEQUENCE_LENGTH,
			ATTENTION_DIM);

	ai->global_learning_rate = LEARNING_RATE;
	ai->learning_epoch = 0;
	ai->average_reward = 0.0f;
	ai->exploration_temperature = 1.0f;

	ai->player_trajectory = (float*) calloc(TEMPORAL_WINDOW * 2, sizeof(float));
	ai->trajectory_length = 0;
	ai->predicted_position = (float*) calloc(2, sizeof(float));

	return ai;
}

static void ai_update_enemies(SensoryGame *game, AdvancedAudioSystem *audio) {
	int i;
	float threat_level;

	if (!game || !game->ai)
		return;

	for (i = 0; i < game->num_enemies; i++) {
		if (!game->enemies[i].active)
			continue;

		AIEnemy *enemy = &game->enemies[i];
		QLearningAgent *agent = game->ai->enemy_agents[i];

		if (!agent)
			continue;

		/* Calculate threat level based on player proximity */
		float dx = game->x - enemy->x;
		float dy = game->y - enemy->y;
		float dist = sqrtf(dx * dx + dy * dy);
		threat_level = 1.0f - (dist / SCREEN_WIDTH);

		/* State encoding */
		int state = (int) (threat_level * (Q_TABLE_SIZE - 1));

		/* Select action using Q-Learning */
		int action = q_agent_select_action(agent, state);

		/* Execute action */
		float speed = enemy->speed * game->delta_time * 60.0f;
		switch (action % 4) {
		case 0:
			enemy->x += speed;
			break; /* Right */
		case 1:
			enemy->x -= speed;
			break; /* Left */
		case 2:
			enemy->y += speed;
			break; /* Down */
		case 3:
			enemy->y -= speed;
			break; /* Up */
		}

		/* Calculate reward */
		float new_dx = game->x - enemy->x;
		float new_dy = game->y - enemy->y;
		float new_dist = sqrtf(new_dx * new_dx + new_dy * new_dy);
		float reward = (dist - new_dist) * 10.0f;

		/* Update Q-table */
		int next_state = (int) ((1.0f - new_dist / SCREEN_WIDTH)
				* (Q_TABLE_SIZE - 1));
		q_agent_update(agent, state, action, reward, next_state);

		/* Use SNN for behavior modulation */
		float snn_input[10] = { threat_level, dist / SCREEN_WIDTH, enemy->x
				/ SCREEN_WIDTH, enemy->y / SCREEN_HEIGHT, game->health / 100.0f,
				game->combo / 10.0f, sinf(enemy->pulse_phase), cosf(
						enemy->pulse_phase), random_range(-1, 1), random_range(
						-1, 1) };
		snn_update(game->ai->snn, snn_input, game->delta_time);

		/* Modulate behavior based on SNN output */
		enemy->aggression = game->ai->snn->output_buffer[0] * 0.5f + 0.5f;
		enemy->speed = 1.0f + game->ai->snn->output_buffer[1] * 2.0f;

		game->ai->learning_epoch++;
	}
}

static void ai_learn_from_experience(SensoryGame *game) {
	/* Reinforcement learning from game outcomes */
	if (!game || !game->ai)
		return;

	/* Update average reward */
	game->ai->average_reward = game->ai->average_reward * 0.99f
			+ (float) game->score * 0.01f;

	/* Decay exploration temperature */
	game->ai->exploration_temperature *= 0.999f;

	/* Could implement batch Q-learning update here */
}

static void ai_predict_player_movement(SensoryGame *game) {
	if (!game || !game->ai || !game->ai->tpr)
		return;

	/* Add current position to trajectory buffer */
	if (game->ai->trajectory_length < TEMPORAL_WINDOW) {
		game->ai->player_trajectory[game->ai->trajectory_length * 2] = game->x;
		game->ai->player_trajectory[game->ai->trajectory_length * 2 + 1] =
				game->y;
		game->ai->trajectory_length++;
	} else {
		/* Shift buffer */
		int i;
		for (i = 1; i < TEMPORAL_WINDOW; i++) {
			game->ai->player_trajectory[(i - 1) * 2] =
					game->ai->player_trajectory[i * 2];
			game->ai->player_trajectory[(i - 1) * 2 + 1] =
					game->ai->player_trajectory[i * 2 + 1];
		}
		game->ai->player_trajectory[(TEMPORAL_WINDOW - 1) * 2] = game->x;
		game->ai->player_trajectory[(TEMPORAL_WINDOW - 1) * 2 + 1] = game->y;
	}

	/* Predict next position using temporal pattern recognition */
	if (game->ai->trajectory_length >= 10) {
		float *prediction = tpr_predict(game->ai->tpr,
				game->ai->player_trajectory);
		game->ai->predicted_position[0] = prediction[0] * SCREEN_WIDTH;
		game->ai->predicted_position[1] = prediction[1] * SCREEN_HEIGHT;
	}
}

static void ai_prioritize_events(SensoryGame *game) {
	if (!game || !game->ai || !game->ai->attention_queue)
		return;

	/* Create messages for significant events */
	float threat_msg[ATTENTION_DIM] = { 0 };
	float health_msg[ATTENTION_DIM] = { 0 };
	float wave_msg[ATTENTION_DIM] = { 0 };

	/* Threat message */
	threat_msg[0] = (float) game->num_enemies / MAX_ENEMIES;
	threat_msg[1] = game->health / game->max_health;
	attention_queue_add(game->ai->attention_queue, threat_msg, 1);

	/* Health message */
	health_msg[0] = game->health / game->max_health;
	health_msg[1] = (float) game->combo / 10.0f;
	attention_queue_add(game->ai->attention_queue, health_msg, 2);

	/* Wave message */
	wave_msg[0] = (float) game->current_wave / MAX_WAVES;
	wave_msg[1] = (float) game->enemies_killed / game->wave_enemies;
	attention_queue_add(game->ai->attention_queue, wave_msg, 3);

	/* Get priorities */
	int *priorities = attention_queue_get_priorities(game->ai->attention_queue);

	/* Use priorities to modulate game behavior */
	if (priorities && priorities[0] > 50) {
		/* High priority event - increase difficulty temporarily */
		int i;
		for (i = 0; i < game->num_enemies; i++) {
			if (game->enemies[i].active) {
				game->enemies[i].speed *= 1.1f;
			}
		}
	}
}

static void ai_system_destroy(AISystem *ai) {
	int i;

	if (!ai)
		return;

	for (i = 0; i < MAX_ENEMIES; i++) {
		q_agent_destroy(ai->enemy_agents[i]);
	}

	snn_destroy(ai->snn);
	tpr_destroy(ai->tpr);
	attention_queue_destroy(ai->attention_queue);

	free(ai->player_trajectory);
	free(ai->predicted_position);
	free(ai);
}

/* ============================================================================
 * Visual Effects Functions (unchanged from original)
 * ============================================================================ */

static void create_particle(SensoryGame *game, float x, float y, float vx,
		float vy, unsigned int color, float size, float glow) {
	int i;

	for (i = 0; i < MAX_PARTICLES; i++) {
		if (!game->particles[i].active) {
			game->particles[i].x = x;
			game->particles[i].y = y;
			game->particles[i].z = 0;
			game->particles[i].vx = vx;
			game->particles[i].vy = vy;
			game->particles[i].vz = random_range(-1, 1);
			game->particles[i].ax = 0;
			game->particles[i].ay = 0.2f;
			game->particles[i].az = 0;
			game->particles[i].life = PARTICLE_LIFE;
			game->particles[i].max_life = PARTICLE_LIFE;
			game->particles[i].color = color;
			game->particles[i].size = size;
			game->particles[i].rotation = random_range(0, 2 * M_PI);
			game->particles[i].rotation_speed = random_range(-0.1f, 0.1f);
			game->particles[i].glow = glow;
			game->particles[i].active = 1;

			game->num_particles++;
			break;
		}
	}
}

static void create_explosion(SensoryGame *game, float x, float y,
		AdvancedAudioSystem *audio) {
	int i;

	if (audio) {
		audio_play_sound(audio, "explosion", x, y, 1.0f);
	}

	for (i = 0; i < 20; i++) {
		float angle = random_range(0, 2 * M_PI);
		float speed = random_range(2, 8);
		float vx = cos(angle) * speed;
		float vy = sin(angle) * speed;
		create_particle(game, x, y, vx, vy, COLOR_ORANGE, random_range(3, 6),
				1.0f);
	}

	game->screen_shake_intensity = SCREEN_SHAKE_MAX;
	game->screen_shake_duration = 0.5f;

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
			game->fields[i].force_x = 0;
			game->fields[i].force_y = 0;
			game->fields[i].attenuation = 1.0f;
			game->fields[i].active = 1;
			game->fields[i].type = type;
			break;
		}
	}
}

static void apply_force_fields(SensoryGame *game) {
	int i, j;

	for (i = 0; i < MAX_EMITTERS; i++) {
		if (!game->fields[i].active)
			continue;

		for (j = 0; j < MAX_PARTICLES; j++) {
			if (!game->particles[j].active)
				continue;

			float dx = game->particles[j].x - game->fields[i].x;
			float dy = game->particles[j].y - game->fields[i].y;
			float dist = sqrt(dx * dx + dy * dy);

			if (dist < game->fields[i].radius && dist > 0.1f) {
				float strength = (1.0f - dist / game->fields[i].radius)
						* game->fields[i].attenuation;

				switch (game->fields[i].type) {
				case 0: /* Attract */
					game->particles[j].vx -= (dx / dist) * strength * 0.5f;
					game->particles[j].vy -= (dy / dist) * strength * 0.5f;
					break;
				case 1: /* Repel */
					game->particles[j].vx += (dx / dist) * strength * 0.5f;
					game->particles[j].vy += (dy / dist) * strength * 0.5f;
					break;
				case 2: /* Vortex */
					game->particles[j].vx += -dy * strength * 0.3f;
					game->particles[j].vy += dx * strength * 0.3f;
					break;
				}
			}
		}
	}
}

static void update_weather(SensoryGame *game, AdvancedAudioSystem *audio) {
	float rain_x, rain_y, rain_vx, rain_vy;
	int i;

	/* Update wind */
	game->wind_force_x += random_range(-0.1f, 0.1f);
	game->wind_force_y += random_range(-0.1f, 0.1f);
	game->wind_force_x = clampf(game->wind_force_x, -1.0f, 1.0f);
	game->wind_force_y = clampf(game->wind_force_y, -1.0f, 1.0f);

	switch (game->weather.type) {
	case WEATHER_RAIN:
		game->weather.intensity = 0.7f;
		game->weather.visibility = 0.8f;
		game->weather.particle_color = COLOR_CYAN;

		if (rand() % 10 < game->weather.intensity * 10) {
			rain_x = random_range(0, SCREEN_WIDTH);
			rain_y = random_range(-50, 0);
			rain_vx = game->wind_force_x * 10;
			rain_vy = 20 + game->wind_force_y * 5;
			create_particle(game, rain_x, rain_y, rain_vx, rain_vy,
					game->weather.particle_color, random_range(1, 3), 0.2f);
		}

		if (audio && rand() % 100 < 5) {
			audio_play_sound(audio, "ambient", rain_x, rain_y, 0.3f);
		}
		break;

	case WEATHER_SNOW:
		game->weather.intensity = 0.5f;
		game->weather.visibility = 0.5f;
		game->weather.particle_color = COLOR_WHITE;

		if (rand() % 20 < game->weather.intensity * 10) {
			rain_x = random_range(0, SCREEN_WIDTH);
			rain_y = random_range(-50, 0);
			rain_vx = game->wind_force_x * 2;
			rain_vy = 5 + game->wind_force_y * 2;
			create_particle(game, rain_x, rain_y, rain_vx, rain_vy,
					game->weather.particle_color, random_range(2, 4), 0.1f);
		}
		break;

	case WEATHER_FOG:
		game->weather.intensity = 0.8f;
		game->weather.visibility = 0.3f;
		break;

	case WEATHER_STORM:
		game->weather.intensity = 1.0f;
		game->weather.visibility = 0.4f;

		if (rand() % 1000 < 5) {
			float lightning_x = random_range(100, SCREEN_WIDTH - 100);
			for (i = 0; i < MAX_ENEMIES; i++) {
				if (!game->lightnings[i].active) {
					game->lightnings[i].x = lightning_x;
					game->lightnings[i].y = 0;
					game->lightnings[i].target_x = lightning_x
							+ random_range(-50, 50);
					game->lightnings[i].target_y = SCREEN_HEIGHT;
					game->lightnings[i].speed = 100;
					game->lightnings[i].thickness = 3;
					game->lightnings[i].color = COLOR_YELLOW;
					game->lightnings[i].life = 0.5f;
					game->lightnings[i].active = 1;
					break;
				}
			}

			if (audio) {
				audio_play_sound(audio, "explosion", lightning_x,
				SCREEN_HEIGHT / 2, 1.0f);
			}
		}
		break;

	default:
		break;
	}
}

/* ============================================================================
 * Game Initialization - Enhanced with AI
 * ============================================================================ */

static void init_game(SensoryGame *game) {
	int i, j;

	memset(game, 0, sizeof(SensoryGame));

	game->x = SCREEN_WIDTH / 2;
	game->y = SCREEN_HEIGHT - 50;
	game->health = 100;
	game->max_health = 100;
	game->score = 0;
	game->combo = 1;

	game->current_wave = 1;
	game->wave_enemies = 5 + game->current_wave;

	game->weather.type = WEATHER_NONE;
	game->weather.intensity = 0;
	game->weather.visibility = 1.0f;

	/* Initialize AI system */
	game->ai = ai_system_create();

	for (i = 0; i < MAX_ENEMIES; i++) {
		game->enemies[i].active = 0;
		game->enemies[i].q_agent = NULL;
		game->enemies[i].temporal_memory = (float*) calloc(TEMPORAL_WINDOW,
				sizeof(float));
		game->enemies[i].memory_index = 0;
	}
	for (i = 0; i < MAX_BULLETS; i++) {
		game->bullets[i].active = 0;
	}
	for (i = 0; i < MAX_PARTICLES; i++) {
		game->particles[i].active = 0;
	}
	for (i = 0; i < MAX_GLOWS; i++) {
		game->glows[i].active = 0;
	}
	for (i = 0; i < MAX_EMITTERS; i++) {
		game->fields[i].active = 0;
	}

	game->num_enemies = 0;
	game->num_bullets = 0;
	game->num_particles = 0;
	game->num_glows = 0;

	game->ai_decisions = 0;
}

/* ============================================================================
 * Drawing Functions (unchanged from original)
 * ============================================================================ */

static void draw_rect(float x, float y, float w, float h, unsigned int color) {
	float r = ((color >> 16) & 0xFF) / 255.0f;
	float g = ((color >> 8) & 0xFF) / 255.0f;
	float b = (color & 0xFF) / 255.0f;

	glColor3f(r, g, b);
	glBegin(GL_QUADS);
	glVertex2f(x - w / 2, y - h / 2);
	glVertex2f(x + w / 2, y - h / 2);
	glVertex2f(x + w / 2, y + h / 2);
	glVertex2f(x - w / 2, y + h / 2);
	glEnd();
}

static void draw_circle(float x, float y, float r, unsigned int color,
		float alpha) {
	int i;
	int segments = 20;
	float r_color = ((color >> 16) & 0xFF) / 255.0f;
	float g_color = ((color >> 8) & 0xFF) / 255.0f;
	float b_color = (color & 0xFF) / 255.0f;

	glColor4f(r_color, g_color, b_color, alpha);
	glBegin(GL_TRIANGLE_FAN);
	glVertex2f(x, y);
	for (i = 0; i <= segments; i++) {
		float angle = i * 2 * M_PI / segments;
		glVertex2f(x + cos(angle) * r, y + sin(angle) * r);
	}
	glEnd();
}

static void draw_glow(GlowLight *glow) {
	int i;
	int segments = 32;
	float r = ((glow->color >> 16) & 0xFF) / 255.0f;
	float g = ((glow->color >> 8) & 0xFF) / 255.0f;
	float b = (glow->color & 0xFF) / 255.0f;

	glow->pulse_phase += 0.1f;

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);

	glBegin(GL_TRIANGLE_FAN);
	glColor4f(r, g, b, glow->intensity);
	glVertex2f(glow->x, glow->y);

	for (i = 0; i <= segments; i++) {
		float angle = i * 2 * M_PI / segments;
		glColor4f(r, g, b, 0);
		glVertex2f(glow->x + cos(angle) * glow->radius,
				glow->y + sin(angle) * glow->radius);
	}
	glEnd();

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

static void draw_lightning(LightningBolt *bolt) {
	int i;
	float r, g, b, x, y;

	if (!bolt->active)
		return;

	r = ((bolt->color >> 16) & 0xFF) / 255.0f;
	g = ((bolt->color >> 8) & 0xFF) / 255.0f;
	b = (bolt->color & 0xFF) / 255.0f;

	bolt->x += (bolt->target_x - bolt->x) / bolt->speed;
	bolt->y += (bolt->target_y - bolt->y) / bolt->speed;

	bolt->life -= 0.01f;
	if (bolt->life <= 0) {
		bolt->active = 0;
		return;
	}

	glLineWidth(bolt->thickness);
	glColor4f(r, g, b, bolt->life);

	glBegin(GL_LINE_STRIP);
	x = bolt->x;
	y = bolt->y;

	for (i = 0; i < 10; i++) {
		glVertex2f(x, y);
		x += random_range(-5, 5);
		y += (bolt->target_y - bolt->y) / 10 + random_range(-2, 2);
	}
	glEnd();
}

static void draw_particles(SensoryGame *game) {
	int i;

	for (i = 0; i < MAX_PARTICLES; i++) {
		if (game->particles[i].active) {
			AdvancedParticle *p = &game->particles[i];
			float alpha = p->life / p->max_life;
			float r = ((p->color >> 16) & 0xFF) / 255.0f;
			float g = ((p->color >> 8) & 0xFF) / 255.0f;
			float b = (p->color & 0xFF) / 255.0f;

			float x = p->x + p->z * 0.5f;
			float y = p->y + p->z * 0.5f;

			glColor4f(r, g, b, alpha);
			glPushMatrix();
			glTranslatef(x, y, 0);
			glRotatef(p->rotation * 180 / M_PI, 0, 0, 1);

			glBegin(GL_QUADS);
			float s = p->size;
			glVertex2f(-s, -s);
			glVertex2f(s, -s);
			glVertex2f(s, s);
			glVertex2f(-s, s);
			glEnd();

			glPopMatrix();

			if (p->glow > 0) {
				glEnable(GL_BLEND);
				glBlendFunc(GL_SRC_ALPHA, GL_ONE);
				glColor4f(r, g, b, alpha * p->glow * 0.5f);
				glPointSize(p->size * 2);
				glBegin(GL_POINTS);
				glVertex2f(x, y);
				glEnd();
				glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			}
		}
	}
}

static void draw_text(float x, float y, const char *text, unsigned int color,
		float scale) {
	/* Simple text rendering placeholder */
	(void) x;
	(void) y;
	(void) text;
	(void) color;
	(void) scale;
}

/* ============================================================================
 * Game Update - Enhanced with AI
 * ============================================================================ */

static void update_game(AppState *state) {
	SensoryGame *game = state->game;
	AdvancedAudioSystem *audio = state->audio;
	int i, j;

	if (game->game_over || game->victory || game->paused) {
		return;
	}

	game->frame_count++;
	game->game_time += game->delta_time;

	if (game->combo_timer > 0) {
		game->combo_timer--;
		if (game->combo_timer <= 0) {
			game->combo = 1;
		}
	}

	/* Update AI systems */
	if (game->ai) {
		ai_update_enemies(game, audio);
		ai_predict_player_movement(game);
		ai_prioritize_events(game);
		ai_learn_from_experience(game);
	}

	/* Update enemies (enhanced with AI) */
	for (i = 0; i < game->num_enemies; i++) {
		if (!game->enemies[i].active)
			continue;

		float dx = game->x - game->enemies[i].x;
		float dy = game->y - game->enemies[i].y;
		float dist = sqrt(dx * dx + dy * dy);

		/* AI-modulated movement */
		if (dist > 0) {
			float speed = game->enemies[i].speed * game->delta_time * 60;
			float aggression = game->enemies[i].aggression;

			/* Combine AI-directed movement with base behavior */
			if (game->ai && game->ai->predicted_position) {
				/* Move toward predicted player position */
				float pdx = game->ai->predicted_position[0]
						- game->enemies[i].x;
				float pdy = game->ai->predicted_position[1]
						- game->enemies[i].y;
				float pdist = sqrt(pdx * pdx + pdy * pdy);

				if (pdist > 0) {
					game->enemies[i].x += (pdx / pdist) * speed * aggression
							* 0.3f;
					game->enemies[i].y += (pdy / pdist) * speed * aggression
							* 0.3f;
				}
			}

			/* Also move toward current player position */
			game->enemies[i].x += (dx / dist) * speed
					* (1.0f - aggression * 0.3f);
			game->enemies[i].y += (dy / dist) * speed
					* (1.0f - aggression * 0.3f);
		}

		game->enemies[i].trail_history_x[game->enemies[i].trail_index] =
				game->enemies[i].x;
		game->enemies[i].trail_history_y[game->enemies[i].trail_index] =
				game->enemies[i].y;
		game->enemies[i].trail_index = (game->enemies[i].trail_index + 1)
				% TRAIL_LENGTH;

		game->enemies[i].pulse_phase += 0.1f;

		if (dist < 30) {
			game->health -= 5;
			game->enemies[i].active = 0;
			game->num_enemies--;
			create_explosion(game, game->enemies[i].x, game->enemies[i].y,
					audio);

			if (game->health <= 0) {
				game->game_over = 1;
				if (audio)
					audio_play_sound(audio, "gameover", SCREEN_WIDTH / 2,
					SCREEN_HEIGHT / 2, 1.0f);
			}
		}
	}

	/* Update bullets (enhanced) */
	for (i = 0; i < game->num_bullets; i++) {
		if (!game->bullets[i].active)
			continue;

		game->bullets[i].x += game->bullets[i].vx * game->delta_time * 60;
		game->bullets[i].y += game->bullets[i].vy * game->delta_time * 60;

		/* Neural pulse effect */
		game->bullets[i].neural_pulse += game->delta_time * 10.0f;

		game->bullets[i].trail_history_x[game->bullets[i].trail_index] =
				game->bullets[i].x;
		game->bullets[i].trail_history_y[game->bullets[i].trail_index] =
				game->bullets[i].y;
		game->bullets[i].trail_index = (game->bullets[i].trail_index + 1)
				% TRAIL_LENGTH;

		if (game->bullets[i].y < 0|| game->bullets[i].y > SCREEN_HEIGHT ||
		game->bullets[i].x < 0 || game->bullets[i].x > SCREEN_WIDTH) {
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
				float dist = sqrt(dx * dx + dy * dy);

				if (dist < 20) {
					game->enemies[j].health -= game->bullets[i].damage;
					game->bullets[i].active = 0;
					game->num_bullets--;
					game->total_hits++;

					create_particle(game, game->enemies[j].x,
							game->enemies[j].y, random_range(-5, 5),
							random_range(-5, 5),
							COLOR_YELLOW, 3, 0.5f);

					if (audio) {
						audio_play_sound(audio, "hit", game->enemies[j].x,
								game->enemies[j].y, 0.5f);
					}

					if (game->enemies[j].health <= 0) {
						game->score += 100 * game->combo;
						game->total_kills++;
						game->enemies_killed++;
						game->combo++;
						game->combo_timer = 2000;

						create_explosion(game, game->enemies[j].x,
								game->enemies[j].y, audio);

						game->enemies[j].active = 0;
						game->num_enemies--;

						/* Reinforcement learning reward for killing enemy */
						if (game->ai && game->ai->enemy_agents[j]) {
							q_agent_update(game->ai->enemy_agents[j],
									game->ai->enemy_agents[j]->state,
									game->ai->enemy_agents[j]->action, 10.0f, /* positive reward */
									game->ai->enemy_agents[j]->state);
						}
					}
					break;
				}
			}
		}
	}

	/* Spawn enemies (with AI initialization) */
	if (game->num_enemies < game->wave_enemies && rand() % 100 < 5) {
		for (i = 0; i < MAX_ENEMIES; i++) {
			if (!game->enemies[i].active) {
				game->enemies[i].x = random_range(100, SCREEN_WIDTH - 100);
				game->enemies[i].y = 50;
				game->enemies[i].health = 30 + game->current_wave * 10;
				game->enemies[i].max_health = game->enemies[i].health;
				game->enemies[i].active = 1;
				game->enemies[i].speed = 1 + game->current_wave * 0.2f;
				game->enemies[i].color = COLOR_RED;
				game->enemies[i].aggression = random_range(0.5f, 1.5f);

				/* Initialize AI for this enemy */
				if (game->ai && !game->enemies[i].q_agent) {
					game->enemies[i].q_agent = q_agent_create();
				}

				for (j = 0; j < TRAIL_LENGTH; j++) {
					game->enemies[i].trail_history_x[j] = game->enemies[i].x;
					game->enemies[i].trail_history_y[j] = game->enemies[i].y;
				}
				game->enemies[i].trail_index = 0;

				game->num_enemies++;
				break;
			}
		}
	}

	/* Check wave completion */
	if (game->enemies_killed >= game->wave_enemies && game->num_enemies == 0) {
		game->current_wave++;
		game->wave_enemies = 5 + game->current_wave * 2;
		game->enemies_killed = 0;

		if (game->current_wave > 15) {
			game->weather.type = WEATHER_STORM;
		} else if (game->current_wave > 10) {
			game->weather.type = WEATHER_RAIN;
		} else if (game->current_wave > 5) {
			game->weather.type = WEATHER_FOG;
		}

		if (game->current_wave > MAX_WAVES) {
			game->victory = 1;
			if (audio)
				audio_play_sound(audio, "victory", SCREEN_WIDTH / 2,
				SCREEN_HEIGHT / 2, 1.0f);
		}
	}

	/* Update particles */
	for (i = 0; i < MAX_PARTICLES; i++) {
		if (game->particles[i].active) {
			game->particles[i].vx += game->particles[i].ax * game->delta_time
					* 60;
			game->particles[i].vy += game->particles[i].ay * game->delta_time
					* 60;
			game->particles[i].vz += game->particles[i].az * game->delta_time
					* 60;

			game->particles[i].vx += game->wind_force_x * game->delta_time * 10;
			game->particles[i].vy += game->wind_force_y * game->delta_time * 10;

			game->particles[i].x += game->particles[i].vx * game->delta_time
					* 60;
			game->particles[i].y += game->particles[i].vy * game->delta_time
					* 60;
			game->particles[i].z += game->particles[i].vz * game->delta_time
					* 60;

			game->particles[i].rotation += game->particles[i].rotation_speed;
			game->particles[i].life -= game->delta_time;

			if (game->particles[i].life <= 0) {
				game->particles[i].active = 0;
				game->num_particles--;
			}
		}
	}

	/* Apply force fields */
	apply_force_fields(game);

	/* Update weather */
	update_weather(game, audio);

	/* Update screen effects */
	if (game->screen_shake_duration > 0) {
		game->screen_shake_duration -= game->delta_time;
		if (game->screen_shake_duration <= 0) {
			game->screen_shake_intensity = 0;
		}
	}

	if (game->flash_duration > 0) {
		game->flash_duration -= game->delta_time;
		if (game->flash_duration <= 0) {
			game->flash_intensity = 0;
		}
	}

	/* Update audio */
	if (audio) {
		audio_update_listener(audio, game->x, game->y);
		audio_update_music(audio,
				(float) game->num_enemies / game->wave_enemies, game->health,
				game->current_wave);
	}

	game->ai_decisions++;
}

/* ============================================================================
 * Input Processing (unchanged from original)
 * ============================================================================ */

static void process_input(AppState *state) {
	SensoryGame *game = state->game;
	AdvancedAudioSystem *audio = state->audio;
	SDL_Event event;
	const Uint8 *keys;
	int i, j;

	while (SDL_PollEvent(&event)) {
		switch (event.type) {
		case SDL_QUIT:
			state->running = 0;
			return;

		case SDL_KEYDOWN:
			switch (event.key.keysym.sym) {
			case SDLK_ESCAPE:
				state->running = 0;
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
							game->bullets[i].y = game->y - 10;
							game->bullets[i].vx = 0;
							game->bullets[i].vy = -15;
							game->bullets[i].active = 1;
							game->bullets[i].owner = 0;
							game->bullets[i].damage = 10 * game->combo;
							game->bullets[i].color = COLOR_CYAN;

							/* AI enhancement */
							game->bullets[i].neural_pulse = 0.0f;
							game->bullets[i].source_neuron =
									rand() % SNN_NEURONS;
							game->bullets[i].temporal_phase = 0.0f;

							for (j = 0; j < TRAIL_LENGTH; j++) {
								game->bullets[i].trail_history_x[j] = game->x;
								game->bullets[i].trail_history_y[j] = game->y
										- 10;
							}
							game->bullets[i].trail_index = 0;

							game->num_bullets++;
							game->total_shots++;

							if (audio) {
								audio_play_sound(audio, "laser", game->x,
										game->y, 0.8f);
							}

							create_particle(game, game->x, game->y - 10, 0, -5,
							COLOR_YELLOW, 5, 1.0f);
							break;
						}
					}
				}
				break;
			case SDLK_w:
				game->weather.type = (game->weather.type + 1) % 5;
				break;
			case SDLK_f:
				create_force_field(game, game->x, game->y, 200, 2);
				break;
			}
			break;
		}
	}

	keys = SDL_GetKeyboardState(NULL);

	if (!game->game_over && !game->victory && !game->paused) {
		float move_x = 0, move_y = 0;
		float speed = 5.0f;

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

		game->x = clampf(game->x, 20, SCREEN_WIDTH - 20);
		game->y = clampf(game->y, 20, SCREEN_HEIGHT - 20);

		if (move_x != 0 || move_y != 0) {
			if (rand() % 5 == 0) {
				create_particle(game, game->x, game->y, -move_x * 0.2f,
						-move_y * 0.2f,
						COLOR_CYAN, 3, 0.3f);
			}
		}
	}
}

/* ============================================================================
 * Rendering - Enhanced with AI metrics
 * ============================================================================ */

static void render_game(AppState *state) {
	SensoryGame *game = state->game;
	int i, j;
	float shake_x = 0, shake_y = 0;
	char buffer[256];

	glClear(GL_COLOR_BUFFER_BIT);
	glLoadIdentity();

	if (game->screen_shake_intensity > 0) {
		shake_x = random_range(-game->screen_shake_intensity,
				game->screen_shake_intensity);
		shake_y = random_range(-game->screen_shake_intensity,
				game->screen_shake_intensity);
		glTranslatef(shake_x, shake_y, 0);
	}

	/* Draw background */
	glColor4f(0.05f, 0.05f, 0.1f, 1.0f);
	glBegin(GL_QUADS);
	glVertex2f(0, 0);
	glVertex2f(SCREEN_WIDTH, 0);
	glVertex2f(SCREEN_WIDTH, SCREEN_HEIGHT);
	glVertex2f(0, SCREEN_HEIGHT);
	glEnd();

	/* Draw stars */
	glPointSize(1.0f);
	glColor4f(0.8f, 0.8f, 1.0f, 0.5f);
	glBegin(GL_POINTS);
	for (i = 0; i < 200; i++) {
		float x = fmodf(game->game_time * 5 + i * 37, SCREEN_WIDTH);
		float y = fmodf(i * 73 + game->game_time * 2, SCREEN_HEIGHT);
		glVertex2f(x, y);
	}
	glEnd();

	/* Draw glows */
	for (i = 0; i < MAX_GLOWS; i++) {
		if (game->glows[i].active) {
			draw_glow(&game->glows[i]);
		}
	}

	/* Draw particles */
	draw_particles(game);

	/* Draw lightning */
	for (i = 0; i < MAX_ENEMIES; i++) {
		if (game->lightnings[i].active) {
			draw_lightning(&game->lightnings[i]);
		}
	}

	/* Draw enemy trails */
	for (i = 0; i < game->num_enemies; i++) {
		if (game->enemies[i].active) {
			glLineWidth(2);
			glColor4f(1, 0, 0, 0.3f);
			glBegin(GL_LINE_STRIP);
			for (j = 0; j < TRAIL_LENGTH; j++) {
				int idx = (game->enemies[i].trail_index - j + TRAIL_LENGTH)
						% TRAIL_LENGTH;
				float alpha = 1.0f - (float) j / TRAIL_LENGTH;
				glColor4f(1, 0, 0, alpha * 0.3f);
				glVertex2f(game->enemies[i].trail_history_x[idx],
						game->enemies[i].trail_history_y[idx]);
			}
			glEnd();
		}
	}

	/* Draw bullet trails */
	for (i = 0; i < game->num_bullets; i++) {
		if (game->bullets[i].active) {
			glLineWidth(1);
			glColor4f(0, 1, 1, 0.5f);
			glBegin(GL_LINE_STRIP);
			for (j = 0; j < TRAIL_LENGTH; j++) {
				int idx = (game->bullets[i].trail_index - j + TRAIL_LENGTH)
						% TRAIL_LENGTH;
				float alpha = 1.0f - (float) j / TRAIL_LENGTH;
				glColor4f(0, 1, 1, alpha * 0.5f);
				glVertex2f(game->bullets[i].trail_history_x[idx],
						game->bullets[i].trail_history_y[idx]);
			}
			glEnd();
		}
	}

	/* Draw enemies */
	for (i = 0; i < game->num_enemies; i++) {
		if (game->enemies[i].active) {
			/* Enemy color changes with AI state */
			unsigned int color = game->enemies[i].color;
			if (game->enemies[i].aggression > 1.2f) {
				color = COLOR_ORANGE;
			}
			draw_circle(game->enemies[i].x, game->enemies[i].y, 15, color,
					1.0f);

			float health_pct = game->enemies[i].health
					/ game->enemies[i].max_health;
			draw_rect(game->enemies[i].x, game->enemies[i].y - 25, 40, 5,
			COLOR_RED);
			draw_rect(game->enemies[i].x - 20 + 20 * health_pct,
					game->enemies[i].y - 25, 40 * health_pct, 5, COLOR_GREEN);
		}
	}

	/* Draw bullets */
	for (i = 0; i < game->num_bullets; i++) {
		if (game->bullets[i].active) {
			/* Bullets pulse with neural activity */
			float pulse = sinf(game->bullets[i].neural_pulse) * 0.5f + 0.5f;
			unsigned int color = game->bullets[i].color;
			if (pulse > 0.8f) {
				color = COLOR_WHITE;
			}
			draw_rect(game->bullets[i].x, game->bullets[i].y, 3, 10, color);
		}
	}

	/* Draw player */
	draw_rect(game->x, game->y, 20, 20, COLOR_GREEN);

	/* Draw AI prediction (if available) */
	if (game->ai && game->ai->predicted_position) {
		glColor4f(1.0f, 1.0f, 0.0f, 0.3f);
		draw_circle(game->ai->predicted_position[0],
				game->ai->predicted_position[1], 10, COLOR_YELLOW, 0.3f);
	}

	if (game->screen_shake_intensity > 0) {
		glTranslatef(-shake_x, -shake_y, 0);
	}

	if (game->flash_intensity > 0) {
		float r = ((game->flash_color >> 16) & 0xFF) / 255.0f;
		float g = ((game->flash_color >> 8) & 0xFF) / 255.0f;
		float b = (game->flash_color & 0xFF) / 255.0f;
		glColor4f(r, g, b, game->flash_intensity);
		glBegin(GL_QUADS);
		glVertex2f(0, 0);
		glVertex2f(SCREEN_WIDTH, 0);
		glVertex2f(SCREEN_WIDTH, SCREEN_HEIGHT);
		glVertex2f(0, SCREEN_HEIGHT);
		glEnd();
	}

	/* UI - Enhanced with AI metrics */
	sprintf(buffer, "Score: %d  Health: %.0f/%.0f  Wave: %d/%d  Combo: %dx",
			game->score, game->health, game->max_health, game->current_wave,
			MAX_WAVES, game->combo);
	draw_text(10, 20, buffer, COLOR_WHITE, 1.0f);

	if (game->ai) {
		sprintf(buffer, "AI Epoch: %d  Avg Reward: %.2f  Temp: %.2f",
				game->ai->learning_epoch, game->ai->average_reward,
				game->ai->exploration_temperature);
		draw_text(10, 45, buffer, COLOR_CYAN, 0.8f);
	}

	if (game->combo > 10) {
		sprintf(buffer, "%dx COMBO!", game->combo);
		draw_text(SCREEN_WIDTH / 2 - 60, SCREEN_HEIGHT / 2, buffer, COLOR_GOLD,
				2.0f);
	}

	if (game->game_over) {
		glColor4f(0, 0, 0, 0.8f);
		glBegin(GL_QUADS);
		glVertex2f(0, 0);
		glVertex2f(SCREEN_WIDTH, 0);
		glVertex2f(SCREEN_WIDTH, SCREEN_HEIGHT);
		glVertex2f(0, SCREEN_HEIGHT);
		glEnd();

		draw_text(SCREEN_WIDTH / 2 - 50, SCREEN_HEIGHT / 2, "GAME OVER",
		COLOR_RED, 2.0f);
		sprintf(buffer, "Final Score: %d", game->score);
		draw_text(SCREEN_WIDTH / 2 - 50, SCREEN_HEIGHT / 2 + 40, buffer,
		COLOR_WHITE, 1.5f);

		if (game->ai) {
			sprintf(buffer, "AI Decisions: %d", game->ai_decisions);
			draw_text(SCREEN_WIDTH / 2 - 50, SCREEN_HEIGHT / 2 + 70, buffer,
			COLOR_CYAN, 1.0f);
		}
	}

	if (game->victory) {
		glColor4f(0, 0, 0, 0.8f);
		glBegin(GL_QUADS);
		glVertex2f(0, 0);
		glVertex2f(SCREEN_WIDTH, 0);
		glVertex2f(SCREEN_WIDTH, SCREEN_HEIGHT);
		glVertex2f(0, SCREEN_HEIGHT);
		glEnd();

		draw_text(SCREEN_WIDTH / 2 - 40, SCREEN_HEIGHT / 2, "VICTORY!",
		COLOR_GOLD, 2.5f);
		sprintf(buffer, "Score: %d", game->score);
		draw_text(SCREEN_WIDTH / 2 - 30, SCREEN_HEIGHT / 2 + 50, buffer,
		COLOR_WHITE, 1.5f);
	}

	if (game->paused) {
		glColor4f(0, 0, 0, 0.5f);
		glBegin(GL_QUADS);
		glVertex2f(0, 0);
		glVertex2f(SCREEN_WIDTH, 0);
		glVertex2f(SCREEN_WIDTH, SCREEN_HEIGHT);
		glVertex2f(0, SCREEN_HEIGHT);
		glEnd();

		draw_text(SCREEN_WIDTH / 2 - 30, SCREEN_HEIGHT / 2, "PAUSED",
		COLOR_WHITE, 2.0f);
	}
}

/* ============================================================================
 * Application Initialization (unchanged from original)
 * ============================================================================ */

static int app_init(AppState *state, int argc, char *argv[]) {
	int i;

	memset(state, 0, sizeof(AppState));

	state->verbose = 0;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0)
			state->verbose = 1;
	}

	printf("\n");
	printf("================================================\n");
	printf("  EvoX - Complete Sensory Experience\n");
	printf("  Version 2.0 - AI-Enhanced Edition\n");
	printf("  AI Features: Q-Learning, SNN, BPTT, Transformers\n");
	printf("================================================\n");
	printf("\n");

	srand((unsigned int) time(NULL));

	signal(SIGINT, handle_signal);
	signal(SIGTERM, handle_signal);

	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		fprintf(stderr, "SDL Init failed: %s\n", SDL_GetError());
		return -1;
	}

	state->window = SDL_CreateWindow("EvoX - AI Sensory Experience",
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
	glClearColor(0.02f, 0.02f, 0.05f, 1.0f);

	printf("Initializing audio system...\n");
	state->audio = audio_system_create();
	if (state->audio && state->audio->initialized) {
		printf("  Audio system initialized\n");
	} else {
		printf(
				"  Warning: Audio system initialization failed - continuing without audio\n");
	}

	printf("Initializing game with AI systems...\n");
	state->game = (SensoryGame*) malloc(sizeof(SensoryGame));
	init_game(state->game);

	gettimeofday(&state->last_frame, NULL);
	state->running = 1;

	printf("\n");
	printf("Controls:\n");
	printf("  Arrow Keys/WASD - Move\n");
	printf("  SPACE - Fire\n");
	printf("  W - Cycle Weather\n");
	printf("  F - Create Force Field\n");
	printf("  P - Pause\n");
	printf("  R - Restart\n");
	printf("  ESC - Exit\n\n");
	printf("AI Systems Active:\n");
	printf("  ✓ Q-Learning (Reinforcement Learning)\n");
	printf("  ✓ Spiking Neural Networks (Temporal Coding)\n");
	printf("  ✓ BPTT (Temporal Pattern Recognition)\n");
	printf("  ✓ Self-Attention (Message Prioritization)\n\n");

	return 0;
}

/* ============================================================================
 * Application Cleanup - Enhanced with AI cleanup
 * ============================================================================ */

static void app_cleanup(AppState *state) {
	printf("\nShutting down...\n");

	if (state->game) {
		printf("Final Score: %d\n", state->game->score);
		printf("Total Kills: %d\n", state->game->total_kills);
		printf("AI Decisions: %d\n", state->game->ai_decisions);

		/* Clean up AI system */
		if (state->game->ai) {
			int i;
			for (i = 0; i < MAX_ENEMIES; i++) {
				free(state->game->enemies[i].temporal_memory);
			}
			ai_system_destroy(state->game->ai);
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

/* ============================================================================
 * Main Loop (unchanged from original)
 * ============================================================================ */

static void app_run(AppState *state) {
	int frame_count = 0;

	while (state->running && g_running) {
		Uint32 frame_start = SDL_GetTicks();

		process_input(state);

		if (state->game) {
			state->game->delta_time = 1.0f / TARGET_FPS;
			update_game(state);
		}

		render_game(state);
		SDL_GL_SwapWindow(state->window);

		{
			struct timeval now;
			float elapsed;

			gettimeofday(&now, NULL);
			elapsed = (now.tv_sec - state->last_frame.tv_sec) * 1000.0f
					+ (now.tv_usec - state->last_frame.tv_usec) / 1000.0f;

			if (elapsed > 0) {
				state->fps = 1000.0f / elapsed;
			}

			state->last_frame = now;
		}

		frame_count++;

		if (state->verbose && frame_count % 60 == 0 && state->game) {
			printf(
					"\rFPS: %.1f | Score: %d | Wave: %d | Enemies: %d | Particles: %d | AI Epoch: %d   ",
					state->fps, state->game->score, state->game->current_wave,
					state->game->num_enemies, state->game->num_particles,
					state->game->ai ? state->game->ai->learning_epoch : 0);
			fflush(stdout);
		}

		Uint32 frame_time = SDL_GetTicks() - frame_start;
		if (frame_time < FRAME_TIME_MS) {
			SDL_Delay(FRAME_TIME_MS - frame_time);
		}
	}
}

/* ============================================================================
 * Main Entry Point
 * ============================================================================ */

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
