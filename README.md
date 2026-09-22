### Sawt
- ✨ Sawt is an automated podcast highlights extraction **AI tool** designed specifically for Arabic language podcasts

#### Sawt objectives:
- 📢 Support the Arabic language podcasts by addressing the limited availability of tools that cater specifically to Arabic content
- ⏳ Reduce time and effort for Arabic content creators by enabling quick sharing of short podcast clips
- 🎥 Enable automatic generation of concise highlight clips along with automated labeling to help viewers and creators quickly grasp the content of each segment
- 🔗 Provide a simple and user friendly interface for easy access to key highlights and streamlined user experience

#### Sawt system processing pipeline:
##### 🌍 Language detection:
- The system checks if the uploaded podcast is in Arabic and proceeds only if it is, otherwise it shows a message

##### 📁 Data collection:
- Users upload audio or video files through a web page, these uploaded files serve as the raw data that the system processes and analyzes

##### 📋 Transcription:
- After the podcast file is uploaded, the system checks if its a video, it extracts the audio, then audio is converted into text using a `speech to text API`

##### 🧩 Modelling architecture:
- 🎧 Audio subsystem: once the transcript is generated, this subsystem is responsible for performing detailed audio and text analysis to identify meaningful segments
- 🎥 Video subsystem: after the audio is processed, this subsystem works on analysing the visual content of video podcast, this part of the system helps detect important visual moments
- 🔗 Merge subsystem: after the audio and video parts are processed, this subsystem combines their results to create the final highlight labeled clips for video podcasts  

#### Sawt structure folder:
```
📁 sawt
├── 📁 backendFolder
│   ├── audioSubsystem.py
│   ├── videoSubsystem.py
│   ├── mergeSubsystem.py
│   ├── speechToText.py
│   ├── app.py                       # Main app to run system
│   ├── ffmpeg.exe
│   ├── key.json                     # Google API credentials
│   └── blazeFaceShortRange.tflite   # Lightweight face detection model
├── 📁 assetsFolder
├── 📁 pagesEnglish
├── 📁 pagesArabic
├── mongosh.exe                      # MongoDB shel
```

#### Sawt demo: 
- [Watch the demo video (download)](https://github.com/HananMurrar/Sawt/raw/main/Sawt/Demo.mov)

#### Sawt team:
- Ahmed Naser
- Hanan Murrar
- Leen Issa
