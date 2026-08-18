# Third-Party Notices

Last reviewed: 2026-08-17

## Scope

The project-authored software is licensed under the Business Source License
1.1 in `LICENSE`, with a four-year change to the MIT License for each version.
That license does not replace or override the licenses of third-party
software, model weights, services, media, data, or other materials used by or
stored in this repository. Those materials remain subject to their own terms.

This file records the third-party materials that are material to the project's
current source and default runtime. It must accompany source and binary
distributions of the project. A distributor must also retain any copyright,
license, attribution, source-offer, or `NOTICE` files supplied with the exact
third-party artifacts included in its distribution.

## Important non-commercial model restrictions

The default pipeline uses the DiariZen and DialogueSidon model weights. Both
are offered under Creative Commons Attribution-NonCommercial 4.0 International
(`CC-BY-NC-4.0`). The project's Business Source License does not grant
commercial-use rights for those weights. A deployment using either model must
comply with the non-commercial restriction or obtain separate permission from
the applicable rights holder.

Replacing a restricted model does not by itself establish that the resulting
deployment is compliant. The replacement and all other included dependencies,
models, inputs, and outputs must be reviewed under their own terms.

## Source-code components

### DiariZen

- Component: DiariZen speaker-diarization toolkit
- Upstream: <https://github.com/BUTSpeechFIT/DiariZen>
- Revision: `844f5555b0a98acd0931511fc641a8c5b8ba92c7`
- Used by: `tasks/diarize-audio-part`
- Distribution: retrieved from Git at installation time; not vendored in this
  source repository
- License: MIT
- Copyright: Copyright (c) 2024 BUT Speech@FIT
- Project changes: no upstream source files are modified in this repository;
  project-authored adapter code calls the upstream package

### pyannote.audio fork distributed with DiariZen

- Component: the `pyannote-audio` subdirectory of DiariZen
- Upstream: <https://github.com/BUTSpeechFIT/DiariZen/tree/844f5555b0a98acd0931511fc641a8c5b8ba92c7/pyannote-audio>
- Revision: `844f5555b0a98acd0931511fc641a8c5b8ba92c7`
- Used by: `tasks/diarize-audio-part`
- Distribution: retrieved from Git at installation time; not vendored in this
  source repository
- License: MIT
- Copyright: Copyright (c) 2020 CNRS
- Project changes: no upstream source files are modified in this repository

### pyannote.audio

- Component: pyannote.audio
- Upstream: <https://github.com/pyannote/pyannote-audio>
- Resolved version: `4.0.7` in
  `tasks/split-raw-audio-into-parts/uv.lock`
- Used by: `tasks/split-raw-audio-into-parts`
- Distribution: retrieved from the Python package index at installation time;
  not vendored in this source repository
- License: MIT
- Copyright: Copyright (c) 2020 CNRS
- Project changes: no upstream source files are modified in this repository

The following MIT terms apply to the MIT-licensed components and copyright
notices in this section:

> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

## Model components

### DiariZen WavLM Large S80 MD v2

- Model: `BUT-FIT/diarizen-wavlm-large-s80-md-v2`
- Upstream: <https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md-v2>
- Upstream credit: BUT Speech@FIT and the DiariZen authors
- Used by: `tasks/diarize-audio-part`
- Distribution: downloaded at runtime; weights are not stored in this source
  repository
- License: Creative Commons Attribution-NonCommercial 4.0 International
  (`CC-BY-NC-4.0`)
- License text: <https://creativecommons.org/licenses/by-nc/4.0/legalcode>
- Project changes: the project downloads and runs the upstream weights without
  modifying the weight files
- Restriction: non-commercial use only unless separate permission is obtained

The current configuration does not pin a model-repository revision. The
upstream revision reviewed for this notice was
`f27b9ffbedcf422856d104ecee9b94be37ea578e`. A distributor must verify the
license and notices of the actual revision it downloads.

### DialogueSidon

- Model: `sarulab-speech/DialogueSidon`
- Upstream: <https://huggingface.co/sarulab-speech/DialogueSidon>
- Revision: `d43d7478402a5527136c6733c3f4359c37b312ab`
- Upstream credit: Wataru Nakata, Yuki Saito, Kazuki Yamauchi, Emiru Tsunoo,
  and Hiroshi Saruwatari
- Used by: `tasks/separate-chunk`
- Distribution: downloaded at runtime; weights are not stored in this source
  repository
- License: Creative Commons Attribution-NonCommercial 4.0 International
  (`CC-BY-NC-4.0`)
- License text: <https://creativecommons.org/licenses/by-nc/4.0/legalcode>
- Project changes: the project downloads and runs the upstream weights without
  modifying the weight files
- Restriction: non-commercial use only unless separate permission is obtained

### NVIDIA Parakeet TDT 0.6B v3

- Model: `nvidia/parakeet-tdt-0.6b-v3`
- Upstream: <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3>
- Revision: `541d1f99c6b0c3cd0b11a95167540bb8edefd82b`
- Upstream credit: NVIDIA
- Used by: `tasks/transcribe-chunk`
- Distribution: downloaded at runtime; weights are not stored in this source
  repository
- License: Creative Commons Attribution 4.0 International (`CC-BY-4.0`)
- License text: <https://creativecommons.org/licenses/by/4.0/legalcode>
- Project changes: the project downloads and runs the upstream weights without
  modifying the weight files

### Microsoft WavLM Base Plus for Speaker Verification

- Model: `microsoft/wavlm-base-plus-sv`
- Upstream: <https://huggingface.co/microsoft/wavlm-base-plus-sv>
- Revision: `feb593a6c23c1cc3d9510425c29b0a14d2b07b1e`
- Upstream credit: Microsoft and the WavLM authors listed in the upstream model
  card
- Used by: `tasks/separate-chunk`
- Distribution: downloaded at runtime; weights are not stored in this source
  repository
- License identified by the upstream model card: Creative Commons
  Attribution-ShareAlike 3.0 Unported (`CC-BY-SA-3.0`)
- License text: <https://creativecommons.org/licenses/by-sa/3.0/legalcode>
- Project changes: the project downloads and runs the upstream weights without
  modifying the weight files

### pyannote Segmentation 3.0

- Model: `pyannote/segmentation-3.0`
- Upstream: <https://huggingface.co/pyannote/segmentation-3.0>
- Upstream credit: pyannote and CNRS
- Copyright: Copyright (c) 2023 CNRS
- Used by: `tasks/split-raw-audio-into-parts`
- Distribution: downloaded at runtime; weights are not stored in this source
  repository
- License: MIT
- Project changes: the project downloads and runs the upstream weights without
  modifying the weight files
- Access condition: the upstream repository is gated and requires an
  authenticated user to accept its access conditions

The current configuration does not pin a model-repository revision. The
upstream revision reviewed for this notice was
`e66f3d3b9eb0873085418a7b813d3b369bf160bb`. A distributor must verify the
license and notices of the actual revision it downloads. The MIT terms quoted
in the source-code section above apply to this model's MIT notice.

## Python and system dependencies

The Python dependencies are declared in each component's `pyproject.toml` and
resolved in the adjacent `uv.lock`. These packages and their transitive
dependencies are not relicensed under the project's Business Source License.
The source repository does not vendor their package contents.

An executable, wheel, container image, virtual-machine image, or other binary
distribution may include those packages. Its distributor must produce a
license inventory from the exact installed distributions and include all
license texts, copyright notices, attribution notices, and source-code offers
required by those versions. The lock files are the authoritative version
inventory for each independently deployable project; this notice is not a
substitute for a binary-distribution license bundle.

FFmpeg is an external runtime requirement of the ingest service and is not
included in this source repository. FFmpeg licensing depends on how a
particular build is configured. Anyone distributing an FFmpeg build with this
project must comply with that build's applicable LGPL, GPL, and component
license terms. PostgreSQL, Redis, and S3-compatible services are also external
runtime services and are not distributed by this source repository.

## Bundled NaturalVoices speech/music artifacts

The following model artifacts are stored directly in the source repository:

- `tasks/quality-filter-audio-part/src/voice_pipeline_quality_filter_audio_part/music_artifacts/weights.28-0.13exp1_blstm.hdf5`
- `tasks/quality-filter-audio-part/src/voice_pipeline_quality_filter_audio_part/music_artifacts/mean_gtzan_esc-50_muspeak_musan.npy`
- `tasks/quality-filter-audio-part/src/voice_pipeline_quality_filter_audio_part/music_artifacts/std_gtzan_esc-50_muspeak_musan.npy`

Provenance and license:

- Component: Speech and Music Detection artifacts distributed by NaturalVoices
- NaturalVoices upstream: <https://github.com/3loi/NaturalVoices>
- NaturalVoices revision: `2d816822cc2b45438a8ecc949602f3572aade44b`
- Upstream path: `pipeline_code/music_noise/speech-music-detection/checkpoint/`
- Original component credit: Quentin Lemaire
- Copyright: Copyright (c) 2018 Quentin Lemaire
- License: MIT, as stated in the specific
  `pipeline_code/music_noise/speech-music-detection/LICENSE` file
- License source: <https://github.com/3loi/NaturalVoices/blob/2d816822cc2b45438a8ecc949602f3572aade44b/pipeline_code/music_noise/speech-music-detection/LICENSE>
- Used by: `tasks/quality-filter-audio-part`
- Distribution: the three files are bundled in this source repository
- Project changes: none to the three artifact files; their SHA-256 digests
  exactly match the files at the NaturalVoices revision above

The MIT terms quoted in the source-code section above apply to these artifacts
and must be retained together with the copyright notice. NaturalVoices
describes the artifacts as part of its speech/music detection pipeline. The
associated NaturalVoices publication is "Towards Naturalistic Voice
Conversion: NaturalVoices Dataset with an Automatic Processing Pipeline" by
A. N. Salman, Z. Du, S. S. Chandra, I. R. Ulgen, C. Busso, and B. Sisman,
Interspeech 2024.

## No endorsement

The names of third-party projects, authors, and rights holders are provided
only for attribution. Their inclusion does not imply that they endorse
AveraLabs or this project.
