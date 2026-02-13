import 'dart:io';
import 'package:flutter/material.dart';
import '../db/db_helper.dart';
import '../services/predict_service.dart';
import 'result_screen.dart';

class AnalyzingScreen extends StatefulWidget {
  final String audioPath;
  final String token;

  const AnalyzingScreen({
    super.key,
    required this.audioPath,
    required this.token,
  });

  @override
  State<AnalyzingScreen> createState() => _AnalyzingScreenState();
}

class _AnalyzingScreenState extends State<AnalyzingScreen> {
  String? error;

  @override
  void initState() {
    super.initState();
    _run();
  }

  Future<void> _run() async {
    try {
      final predictService = PredictService();

      final response = await predictService.sendAudioForPrediction(
        audioFile: File(widget.audioPath),
        token: widget.token,
      );

      /// 🔐 Label güvenli alma
      final label =
          response['prediction'] ?? response['label'] ?? 'Bilinmiyor';

      /// 🔐 Confidence güvenli parse
      final rawConfidence = response['confidence'];
      final confidence = rawConfidence is num
          ? rawConfidence.toDouble()
          : double.tryParse(rawConfidence.toString()) ?? 0.0;

      /// 💾 Local DB kayıt
      final rec = AnalysisRecord(
        createdAtIso: DateTime.now().toIso8601String(),
        label: label,
        confidence: confidence,
        audioPath: widget.audioPath,
      );

      await DbHelper.instance.insertRecord(rec);

      if (!mounted) return;

      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) => ResultScreen(
            result: label,
            confidence: confidence,
            audioPath: widget.audioPath,
            createdAtIso: rec.createdAtIso,
            fromHistory: false,
          ),
        ),
      );
    } catch (e) {
      /// 🔐 401 kontrolü (token expired)
      if (e.toString().contains("401")) {
        if (!mounted) return;
        Navigator.popUntil(context, (route) => route.isFirst);
      } else {
        setState(() => error = e.toString());
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _Bg(
        child: SafeArea(
          child: Center(
            child: Padding(
              padding: const EdgeInsets.all(18),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(
                    Icons.memory_rounded,
                    color: Colors.white,
                    size: 60,
                  ),
                  const SizedBox(height: 12),
                  const Text(
                    "Analiz Ediliyor...",
                    style: TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.w900,
                      fontSize: 18,
                    ),
                  ),
                  const SizedBox(height: 20),

                  /// 🔄 Loading
                  if (error == null)
                    const CircularProgressIndicator(
                      color: Colors.white,
                    )

                  /// ❌ Hata Durumu
                  else ...[
                    Text(
                      "API Hatası:\n$error",
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                        color: Colors.yellow,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                    const SizedBox(height: 20),
                    SizedBox(
                      width: double.infinity,
                      height: 50,
                      child: ElevatedButton(
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.white,
                          foregroundColor: const Color(0xFF0B4A7A),
                        ),
                        onPressed: () => Navigator.pop(context),
                        child: const Text(
                          "Geri Dön",
                          style: TextStyle(fontWeight: FontWeight.w900),
                        ),
                      ),
                    )
                  ],
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _Bg extends StatelessWidget {
  final Widget child;
  const _Bg({required this.child});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            Color(0xFF0B4A7A),
            Color(0xFF1565C0),
          ],
        ),
      ),
      child: child,
    );
  }
}


