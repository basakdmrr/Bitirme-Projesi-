import 'dart:async';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:record/record.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
import 'analyzing_screen.dart';
import '../services/auth_service.dart';
import '../widgets/bg.dart';
class RecordScreen extends StatefulWidget {
  const RecordScreen({super.key});

  @override
  State<RecordScreen> createState() => _RecordScreenState();
}

class _RecordScreenState extends State<RecordScreen> {
  final AudioRecorder _recorder = AudioRecorder();

  static const int totalSeconds = 10;
  int sec = 0;
  bool recording = false;

  Timer? _timer;
  String? audioPath;
  String? error;

  @override
  void dispose() {
    _timer?.cancel();
    _recorder.dispose();
    super.dispose();
  }

  // WAV dosya yolu oluştur
  Future<String> _createWavPath() async {
    final dir = await getApplicationDocumentsDirectory();
    return p.join(
      dir.path,
      "breath_${DateTime.now().millisecondsSinceEpoch}.wav",
    );
  }

  // Kayıdı başlat
  Future<void> _startRecording() async {
    try {
      setState(() {
        error = null;
      });

      final mic = await Permission.microphone.request();
      if (!mic.isGranted) {
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

      setState(() {
        recording = true;
        sec = 0;
        audioPath = path;
      });

      _timer?.cancel();
      _timer = Timer.periodic(const Duration(seconds: 1), (_) {
        if (!mounted) return;

        setState(() => sec++);

        if (sec >= totalSeconds) {
          _stopAndAnalyze();
        }
      });
    } catch (e) {
      setState(() => error = "Kayıt başlatılamadı: $e");
    }
  }

  // Kayıdı durdur ve analize geç
  Future<void> _stopAndAnalyze() async {
  try {
    _timer?.cancel();
    await _recorder.stop();

    setState(() {
      recording = false;
    });

    if (!mounted || audioPath == null) return;

    /// 🔐 TOKEN AL
    final tokenObj = AuthService.token;

    if (tokenObj == null) {
      setState(() {
        error = "Oturum bulunamadı. Lütfen tekrar giriş yapın.";
      });
      return;
    }

    Navigator.pushReplacement(
      context,
      MaterialPageRoute(
        builder: (_) => AnalyzingScreen(
          audioPath: audioPath!,
          token: tokenObj.accessToken,
        ),
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
                // ÜST BAR
                Row(
                  children: [
                    IconButton(
                      onPressed: recording ? null : () => Navigator.pop(context),
                      icon: const Icon(Icons.arrow_back_rounded, color: Colors.white),
                    ),
                    const Spacer(),
                    const Text(
                      "Ses Kaydı",
                      style: TextStyle(color: Colors.white, fontWeight: FontWeight.w900),
                    ),
                    const Spacer(),
                    const SizedBox(width: 48),
                  ],
                ),

                const Spacer(),

                // ANA İKON
                Icon(
                  recording ? Icons.graphic_eq_rounded : Icons.mic_none_rounded,
                  size: 120,
                  color: Colors.white,
                ),

                const SizedBox(height: 16),

                // AÇIKLAMA
                Text(
                  recording
                      ? "Lütfen nefes verin...\nKayıt alınıyor."
                      : "Hazır olduğunda kaydı başlat.",
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.w700,
                    height: 1.3,
                  ),
                ),

                const SizedBox(height: 14),

                // SÜRE
                Text(
                  "$sec / $totalSeconds sn",
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.w900,
                    fontSize: 18,
                  ),
                ),

                const SizedBox(height: 10),

                // PROGRESS BAR
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

                // HATA
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

                // ANA BUTON
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
                    onPressed: recording ? _stopAndAnalyze : _startRecording,
                    child: Text(
                      recording ? "Kaydı Bitir" : "Kaydı Başlat",
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
