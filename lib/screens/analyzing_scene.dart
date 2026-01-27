import 'dart:io';
import 'package:flutter/material.dart';

import '../services/predict_service.dart';
import '../local/db_helper.dart';
import '../local/analysis_record.dart';
import '../screens/result_screen.dart';
import '../widgets/bg.dart';

class AnalyzingScreen extends StatefulWidget {
  final String audioPath;
  const AnalyzingScreen({super.key, required this.audioPath});

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

      final result =
          await predictService.predictWav(File(widget.audioPath));

      final record = AnalysisRecord(
        createdAtIso: DateTime.now().toIso8601String(),
        label: result.label,
        confidence: result.confidence,
        audioPath: widget.audioPath,
      );

      await DbHelper.instance.insertRecord(record);

      if (!mounted) return;
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) => ResultScreen(
            result: record.label,
            confidence: record.confidence,
            audioPath: record.audioPath,
            createdAtIso: record.createdAtIso,
            fromHistory: false,
          ),
        ),
      );
    } catch (e) {
      setState(() => error = e.toString());
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Bg(
        child: SafeArea(
          child: Center(
            child: Padding(
              padding: const EdgeInsets.all(18),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.memory_rounded,
                      color: Colors.white, size: 60),
                  const SizedBox(height: 12),
                  const Text(
                    "Analiz Ediliyor...",
                    style: TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.w900,
                      fontSize: 18,
                    ),
                  ),
                  const SizedBox(height: 10),
                  if (error == null)
                    SizedBox(
                      width: 240,
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(999),
                        child: LinearProgressIndicator(
                          minHeight: 10,
                          backgroundColor:
                              Colors.white.withOpacity(0.25),
                          valueColor:
                              const AlwaysStoppedAnimation<Color>(
                                  Colors.white),
                        ),
                      ),
                    )
                  else ...[
                    const SizedBox(height: 12),
                    Text(
                      "API Hatası:\n$error",
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                        color: Colors.yellow,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                    const SizedBox(height: 12),
                    SizedBox(
                      width: double.infinity,
                      height: 50,
                      child: ElevatedButton(
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.white,
                          foregroundColor:
                              const Color(0xFF0B4A7A),
                        ),
                        onPressed: () => Navigator.pop(context),
                        child: const Text(
                          "Geri Dön",
                          style:
                              TextStyle(fontWeight: FontWeight.w900),
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
