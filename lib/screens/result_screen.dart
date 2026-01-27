import 'package:flutter/material.dart';

import '../widgets/bg.dart';
import '../screens/history_screen.dart';
import '../screens/role_select_screen.dart';

class ResultScreen extends StatelessWidget {
  final String result;
  final double confidence;
  final String audioPath;
  final String createdAtIso;
  final bool fromHistory;

  const ResultScreen({
    super.key,
    required this.result,
    required this.confidence,
    required this.audioPath,
    required this.createdAtIso,
    required this.fromHistory,
  });

  @override
  Widget build(BuildContext context) {
    final percent = (confidence * 100).round();

    final ok =
        result.toLowerCase().contains("healthy") ||
        result.toLowerCase().contains("sağ");

    final safeDate = createdAtIso.length >= 19
        ? createdAtIso.substring(0, 19).replaceAll('T', ' ')
        : createdAtIso;

    return Scaffold(
      body: Bg(
        light: true,
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(18),
            child: Column(
              children: [
                Row(
                  children: [
                    IconButton(
                      onPressed: () => Navigator.pop(context),
                      icon: const Icon(Icons.arrow_back_rounded),
                    ),
                    const Spacer(),
                    const Text(
                      "Sonuç",
                      style: TextStyle(
                        fontWeight: FontWeight.w900,
                        fontSize: 18,
                      ),
                    ),
                    const Spacer(),
                    IconButton(
                      onPressed: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (_) => const HistoryScreen(),
                          ),
                        );
                      },
                      icon: const Icon(Icons.history_rounded),
                    ),
                  ],
                ),
                const Spacer(),
                CircleAvatar(
                  radius: 46,
                  backgroundColor:
                      ok ? const Color(0xFF10B981) : const Color(0xFFF59E0B),
                  child: Icon(
                    ok ? Icons.check_rounded : Icons.warning_rounded,
                    color: Colors.white,
                    size: 56,
                  ),
                ),
                const SizedBox(height: 14),
                Text(
                  "Tahmin: $result",
                  style: const TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.w900,
                  ),
                ),
                const SizedBox(height: 6),
                Text(
                  "Güven Skoru: %$percent",
                  style: TextStyle(
                    color: Colors.black.withOpacity(0.65),
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 10),
                Text(
                  "Tarih: $safeDate",
                  style: TextStyle(
                    color: Colors.black.withOpacity(0.55),
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 14),
                SizedBox(
                  width: double.infinity,
                  height: 54,
                  child: ElevatedButton(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF0EA5E9),
                      foregroundColor: Colors.white,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(16),
                      ),
                    ),
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (_) => const HistoryScreen(),
                        ),
                      );
                    },
                    child: const Text(
                      "Analiz Geçmişi",
                      style: TextStyle(fontWeight: FontWeight.w900),
                    ),
                  ),
                ),
                const SizedBox(height: 10),
                SizedBox(
                  width: double.infinity,
                  height: 54,
                  child: OutlinedButton(
                    style: OutlinedButton.styleFrom(
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(16),
                      ),
                    ),
                    onPressed: () {
                      Navigator.pushAndRemoveUntil(
                        context,
                        MaterialPageRoute(
                          builder: (_) => const RoleSelectScreen(),
                        ),
                        (_) => false,
                      );
                    },
                    child: const Text(
                      "Başa Dön",
                      style: TextStyle(fontWeight: FontWeight.w900),
                    ),
                  ),
                ),
                const SizedBox(height: 10),
                Text(
                  "Not: Bu bir ön değerlendirmedir, kesin tanı için hekime başvurun.",
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    color: Colors.black.withOpacity(0.55),
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const Spacer(),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
