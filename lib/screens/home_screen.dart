import 'package:flutter/material.dart';

import '../widgets/bg.dart';
import '../database/db_helper.dart';
import '../models/analysis_record.dart';
import '../screens/history_screen.dart';
import '../screens/record_screen.dart';
import '../screens/result_screen.dart';
import '../screens/role_select_screen.dart';

class _HomeScreenState extends State<HomeScreen> {
  AnalysisRecord? last;

  @override
  void initState() {
    super.initState();
    _loadLast();
  }

  Future<void> _loadLast() async {
    final items = await DbHelper.instance.getRecords();
    if (!mounted) return;
    setState(() => last = items.isEmpty ? null : items.first);
  }

  Future<void> _goRecord() async {
    await Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => const RecordScreen()),
    );
    await _loadLast();
  }

  Future<void> _goHistory() async {
    await Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => const HistoryScreen()),
    );
    await _loadLast();
  }

  @override
  Widget build(BuildContext context) {
    // 👈 UI AYNEN KALIYOR

  

  @override
  Widget build(BuildContext context) {
    final letter = widget.role.isNotEmpty ? widget.role.characters.first : "U";

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
                    Expanded(
                      child: Text(
                        "Merhaba,\n${widget.userName}",
                        style: const TextStyle(fontSize: 22, fontWeight: FontWeight.w900),
                      ),
                    ),
                    CircleAvatar(
                      backgroundColor: Colors.white.withOpacity(0.7),
                      child: Text(letter, style: const TextStyle(fontWeight: FontWeight.w900)),
                    ),
                  ],
                ),
                const SizedBox(height: 18),
                InkWell(
                  borderRadius: BorderRadius.circular(26),
                  onTap: _goRecord,
                  child: Container(
                    width: double.infinity,
                    height: 170,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(26),
                      color: Colors.white.withOpacity(0.88),
                      boxShadow: [BoxShadow(blurRadius: 20, offset: const Offset(0, 12), color: Colors.black.withOpacity(0.10))],
                    ),
                    child: const Center(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          CircleAvatar(
                            radius: 36,
                            backgroundColor: Color(0xFF14B8A6),
                            child: Icon(Icons.mic_rounded, color: Colors.white, size: 34),
                          ),
                          SizedBox(height: 10),
                          Text("Nefes Kaydına\nBaşla", textAlign: TextAlign.center, style: TextStyle(fontSize: 18, fontWeight: FontWeight.w900)),
                        ],
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 14),
                Row(
                  children: [
                    Expanded(
                      child: _SmallCard(
                        title: "Son Analiz",
                        subtitle: last == null ? "Yok" : last!.label,
                        icon: Icons.check_circle_rounded,
                        onTap: () async {
                          if (last == null) {
                            ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Henüz analiz yok.")));
                            return;
                          }
                          await Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (_) => ResultScreen(
                                result: last!.label,
                                confidence: last!.confidence,
                                audioPath: last!.audioPath,
                                createdAtIso: last!.createdAtIso,
                                fromHistory: true,
                              ),
                            ),
                          );
                          await _loadLast();
                        },
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: _SmallCard(
                        title: "Analiz Geçmişi",
                        subtitle: "Kayıtlar",
                        icon: Icons.history_rounded,
                        onTap: _goHistory,
                      ),
                    ),
                  ],
                ),
                const Spacer(),
                SizedBox(
                  width: double.infinity,
                  height: 54,
                  child: ElevatedButton(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF0EA5E9),
                      foregroundColor: Colors.white,
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                    ),
                    onPressed: () {
                      Navigator.pushReplacement(context, MaterialPageRoute(builder: (_) => const RoleSelectScreen()));
                    },
                    child: const Text("Rol Değiştir", style: TextStyle(fontWeight: FontWeight.w900)),
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
}
class _SmallCard extends StatelessWidget {
  final String title;
  final String subtitle;
  final IconData icon;
  final VoidCallback onTap;
  const _SmallCard({required this.title, required this.subtitle, required this.icon, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return InkWell(
      borderRadius: BorderRadius.circular(18),
      onTap: onTap,
      child: Container(
        height: 120,
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.88),
          borderRadius: BorderRadius.circular(18),
          boxShadow: [BoxShadow(blurRadius: 18, offset: const Offset(0, 10), color: Colors.black.withOpacity(0.08))],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(icon, size: 26, color: const Color(0xFF0EA5E9)),
            const Spacer(),
            Text(title, style: const TextStyle(fontWeight: FontWeight.w900)),
            const SizedBox(height: 2),
            Text(subtitle, style: TextStyle(color: Colors.black.withOpacity(0.6), fontWeight: FontWeight.w700)),
          ],
        ),
      ),
    );
  }
}
