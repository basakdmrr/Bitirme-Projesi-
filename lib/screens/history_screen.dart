import 'package:flutter/material.dart';

import '../widgets/bg.dart';
import '../database/db_helper.dart';
import '../models/analysis_record.dart';
import '../screens/result_screen.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  List<AnalysisRecord> items = [];
  bool loading = true;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final list = await DbHelper.instance.getRecords();
    if (!mounted) return;
    setState(() {
      items = list;
      loading = false;
    });
  }

  Future<void> _clearAll() async {
    await DbHelper.instance.clearAll();
    await _load();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          "Analiz Geçmişi",
          style: TextStyle(fontWeight: FontWeight.w900),
        ),
        actions: [
          IconButton(
            onPressed: _clearAll,
            icon: const Icon(Icons.delete_forever_rounded),
            tooltip: "Tüm kayıtları sil",
          ),
        ],
      ),
      body: Bg(
        light: true,
        child: loading
            ? const Center(child: CircularProgressIndicator())
            : items.isEmpty
                ? const Center(
                    child: Text(
                      "Kayıt yok",
                      style: TextStyle(fontWeight: FontWeight.w900),
                    ),
                  )
                : ListView.separated(
                    padding: const EdgeInsets.all(18),
                    itemCount: items.length,
                    separatorBuilder: (_, __) => const SizedBox(height: 12),
                    itemBuilder: (_, i) {
                      final r = items[i];

                      final date = r.createdAtIso.length >= 19
                          ? r.createdAtIso
                              .substring(0, 19)
                              .replaceAll('T', ' ')
                          : r.createdAtIso;

                      final percent = (r.confidence * 100).round();

                      final color =
                          r.label.toLowerCase().contains("healthy") ||
                                  r.label.toLowerCase().contains("sağ")
                              ? const Color(0xFF10B981)
                              : const Color(0xFFF59E0B);

                      return InkWell(
                        borderRadius: BorderRadius.circular(16),
                        onTap: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (_) => ResultScreen(
                                result: r.label,
                                confidence: r.confidence,
                                audioPath: r.audioPath,
                                createdAtIso: r.createdAtIso,
                                fromHistory: true,
                              ),
                            ),
                          );
                        },
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 14,
                            vertical: 12,
                          ),
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.90),
                            borderRadius: BorderRadius.circular(16),
                            boxShadow: [
                              BoxShadow(
                                blurRadius: 14,
                                offset: const Offset(0, 8),
                                color: Colors.black.withOpacity(0.08),
                              ),
                            ],
                          ),
                          child: Row(
                            children: [
                              CircleAvatar(
                                backgroundColor: color.withOpacity(0.15),
                                child: Icon(
                                  Icons.monitor_heart_rounded,
                                  color: color,
                                ),
                              ),
                              const SizedBox(width: 12),
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      date,
                                      style: const TextStyle(
                                        fontWeight: FontWeight.w900,
                                      ),
                                    ),
                                    const SizedBox(height: 2),
                                    Text(
                                      "${r.label}  •  %$percent",
                                      style: TextStyle(
                                        color: color,
                                        fontWeight: FontWeight.w900,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                              const Icon(Icons.chevron_right_rounded),
                            ],
                          ),
                        ),
                      );
                    },
                  ),
      ),
    );
  }
}
