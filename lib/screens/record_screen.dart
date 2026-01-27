import 'dart:async';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
import 'package:record/record.dart';

import '../widgets/bg.dart';
import 'analyzing_screen.dart';

class RecordScreen extends StatefulWidget {
  const RecordScreen({super.key});

  @override
  State<RecordScreen> createState() => _RecordScreenState();
}

class _RecordScreenState extends State<RecordScreen> {
  final AudioRecorder _recorder = AudioRecorder();

  static const int totalSeconds = 10;

  bool recording = false;
  int sec = 0;
  Timer? t;

  String? audioPath;
  String? error;

  @override
  void dispose() {
    t?.cancel();
    _recorder.dispose();
    super.dispose();
  }

  Future<String> _createWavPath() async {
    final dir = await getApplicationDocumentsDirectory();
    return p.join(
      dir.path,
      "breath_${DateTime.now().millisecondsSinceEpoch}.wav",
    );
  }

  Future<void> _start() async {
    try {
      setState(() => error = null);

      final micStatus = await Permission.microphone.request();
      if (!micStatus.isGranted) {
        setState(() => error = "Mikrofon izni verilmedi.");
        return;
      }

      final path = await _createWavPath();

      await _recorder.start(
        RecordConfig(
          encoder: AudioEncoder.wav,
          sampleRate: 16000,
          numChannels: 1,
          bitRate: 128000,
        ),
        path: path,
      );

      audioPath = path;
      recording = true;
      sec = 0;

      t?.cancel();
      t = Timer.periodic(const Duration(seconds: 1), (_) async {
        if (!mounted) return;

        setState(() => sec++);

        if (sec >= totalSeconds) {
          await _stopAndGoAnalyze();
        }
      });

      setState(() {});
    } catch (e) {
      setState(() => error = "Kayıt başlatılamadı: $e");
    }
  }

  Future<void> _stopAndGoAnalyze() async {
    try {
      t?.cancel();

      if (recording) {
        await _recorder.stop();
        recording = false;
      }

      if (!mounted) return;

      if (audioPath == null) {
        setState(() => error = "Ses dosyası oluşmadı.");
        return;
      }

      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) => AnalyzingScreen(audioPath: audioPath!),
        ),
      );
    } catch (e) {
      setState(() => error = "Kayıt durdurulamadı: $e");
    }
  }

  @override
  Widget build(BuildContext context) {
    final progress = (sec / totalSeconds).clamp(0.0, 1.0);

    return Scaffold(
      body: Bg(
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(18),
            child: Column(
              children: [
                Row(
                  children: [
                    IconButton(
                      onPressed: () => Navigator.pop(context),
                      icon: const Icon(Icons.arrow_back_rounded, color: Colors.white),
                    ),
                    const Spacer(),
                    const Text(
                      "Kayıt",
                      style: TextStyle(color: Colors.white, fontWeight: FontWeight.w900),
                    ),
                    const Spacer(),
                    const SizedBox(width: 48),
                  ],
                ),
                const Spacer(),
                Icon(
                  recording ? Icons.graphic_eq_rounded : Icons.mic_none_rounded,
                  size: 120,
                  color: Colors.white,
                ),
                const SizedBox(height: 14),
                Text(
                  recording
                      ? "Lütfen nefes verin...\nKayıt alınıyor."
                      : "Hazır olduğunda kaydı başlat.",
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.w700,
                    height: 1.2,
                  ),
                ),
                const SizedBox(height: 14),
                Text(
                  "$sec / $totalSeconds sn",
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.w900,
                    fontSize: 18,
                  ),
                ),
                const SizedBox(height: 10),
                ClipRRect(
                  borderRadius: BorderRadius.circular(999),
                  child: LinearProgressIndicator(
                    value: progress,
                    minHeight: 10,
                    backgroundColor: Colors.white.withOpacity(0.25),
                    valueColor: const AlwaysStoppedAnimation<Color>(Colors.white),
                  ),
                ),
                const SizedBox(height: 14),
                if (error != null)
                  Text(
                    error!,
                    textAlign: TextAlign.center,
                    style: const TextStyle(
                      color: Colors.yellow,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                const Spacer(),
                SizedBox(
                  width: double.infinity,
                  height: 54,
                  child: ElevatedButton(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.white,
                      foregroundColor: const Color(0xFF0B4A7A),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(16),
                      ),
                    ),
                    onPressed: recording ? _stopAndGoAnalyze : _start,
                    child: Text(
                      recording ? "Kaydı Bitir & Analiz Et" : "Kaydı Başlat",
                      style: const TextStyle(fontWeight: FontWeight.w900),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
