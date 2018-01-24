import argparse

import pickle
import pydub
from midi2audio import FluidSynth

from DeepBach.data_utils import BACH_DATASET
from DeepBach.model_manager import generation, load_models

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps',
                        help="model's range (default: %(default)s)",
                        type=int, default=16)
    parser.add_argument('-i', '--num_iterations',
                        help='number of gibbs iterations (default: %(default)s)',
                        type=int, default=20000)
    parser.add_argument('-p', '--parallel', nargs='?',
                        help='number of parallel updates (default: 16)',
                        type=int, const=16, default=1)
    parser.add_argument('-l', '--length',
                        help='length of unconstrained generation',
                        type=int, default=160)
    args = parser.parse_args()

    X, X_metadatas, voice_ids, index2notes, note2indexes, metadatas = pickle.load(open(BACH_DATASET, 'rb'))
    NUM_VOICES = len(voice_ids)
    model_name = 'deepbach'
    sequence_length = args.length
    batch_size_per_voice = args.parallel
    midi_out = 'tmp/out.midi'
    wav_out = 'tmp/out.wav'

    iteration = 0
    reharmonization = 0

    num_iterations = args.num_iterations // batch_size_per_voice // (NUM_VOICES - 1)
    parallel = batch_size_per_voice > 1

    models = load_models(model_name, num_voices=NUM_VOICES)
    temperature = 1.
    timesteps = int(models[0].input[0]._keras_shape[1])

    while True:
        print("-- ITERATION %d --" % iteration)
        print("-- REHARMONIZATION %d --" % reharmonization)

        melody = X[reharmonization][0, :]
        chorale_metas = X_metadatas[reharmonization]
        reharmonization = (reharmonization + 1) % len(X)

        generation(model_base_name=model_name, models=models,
                   timesteps=timesteps,
                   melody=melody, initial_seq=None, temperature=temperature,
                   chorale_metas=chorale_metas, parallel=parallel,
                   batch_size_per_voice=batch_size_per_voice,
                   num_iterations=num_iterations,
                   sequence_length=sequence_length,
                   scale=2.0,
                   show=False,
                   output_file=midi_out,
                   pickled_dataset=BACH_DATASET)

        fs = FluidSynth('steinway_grand.sf2')
        fs.midi_to_audio(midi_out, wav_out)
        pydub.AudioSegment.from_wav(wav_out).fade_in(3000).fade_out(3000).export('audio_db/%d.mp3' % iteration)

        iteration = (iteration + 1)  # % 100
