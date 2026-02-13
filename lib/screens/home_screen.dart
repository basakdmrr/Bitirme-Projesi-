import 'package:flutter/material.dart';
import '../db/db_helper.dart';
import '../services/auth_service.dart'; // Token almak için
import 'history_screen.dart';
import 'result_screen.dart';
import 'role_select.dart';
import 'record_screen.dart';
import '../widgets/bg.dart';

class HomeScreen extends StatefulWidget {
  final String role;
  final String userName;
  const HomeScreen({super.key, required this.role, required this.userName});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  AnalysisRecord? last;
  final _authService = AuthService();

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

 Future<void> _goHistory() async {
  final tokenObj = AuthService.token;

  if (tokenObj == null) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text("Token bulunamadı. Lütfen tekrar giriş yapın."),
      ),
    );
    return;
  }

  await Navigator.push(
    context,
    MaterialPageRoute(
      builder: (_) => HistoryScreen(
        token: tokenObj.accessToken,
      ),
    ),
  );

  await _loadLast();
}


  Future<void> _goRecord() async {
    await Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => const RecordScreen()),
    );
    await _loadLast();
  }

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
                        style: const TextStyle(
                          fontSize: 22,
                          fontWeight: FontWeight.w900,
                        ),
                      ),
                    ),
                    CircleAvatar(
                      backgroundColor: Colors.white.withOpacity(0.7),
                      child: Text(
                        letter,
                        style: const TextStyle(fontWeight: FontWeight.w900),
                      ),
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
                      boxShadow: [
                        BoxShadow(
                          blurRadius: 20,
                          offset: const Offset(0, 12),
                          color: Colors.black.withOpacity(0.10),
                        )
                      ],
                    ),
                    child: const Center(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          CircleAvatar(
                            radius: 36,
                            backgroundColor: Color(0xFF14B8A6),
                            child: Icon(
                              Icons.mic_rounded,
                              color: Colors.white,
                              size: 34,
                            ),
                          ),
                          SizedBox(height: 10),
                          Text(
                            "Nefes Kaydına\nBaşla",
                            textAlign: TextAlign.center,
                            style: TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.w900,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 14),
                Row(
                  children: [
                    Expanded(
                      child: SmallCard(
                        title: "Son Analiz",
                        subtitle: last == null ? "Yok" : last!.label,
                        icon: Icons.check_circle_rounded,
                        onTap: () async {
                          if (last == null) {
                            ScaffoldMessenger.of(context).showSnackBar(
                              const SnackBar(
                                content: Text("Henüz analiz yok."),
                              ),
                            );
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
                      child: SmallCard(
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
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(16),
                      ),
                    ),
                    onPressed: () {
                      Navigator.pushReplacement(
                        context,
                        MaterialPageRoute(
                          builder: (_) => const RoleSelectScreen(),
                        ),
                      );
                    },
                    child: const Text(
                      "Rol Değiştir",
                      style: TextStyle(fontWeight: FontWeight.w900),
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

class SmallCard extends StatelessWidget {
  final String title;
  final String subtitle;
  final IconData icon;
  final VoidCallback onTap;

  const SmallCard({
    super.key,
    required this.title,
    required this.subtitle,
    required this.icon,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      borderRadius: BorderRadius.circular(18),
      onTap: onTap,
      child: Container(
        height: 120,
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(18),
          color: Colors.white.withOpacity(0.88),
          boxShadow: [
            BoxShadow(
              blurRadius: 15,
              offset: const Offset(0, 8),
              color: Colors.black.withOpacity(0.08),
            )
          ],
        ),
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(icon, color: const Color(0xFF14B8A6), size: 28),
            const Spacer(),
            Text(
              title,
              style: const TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w600,
                color: Colors.black54,
              ),
            ),
            const SizedBox(height: 2),
            Text(
              subtitle,
              style: const TextStyle(
                fontSize: 15,
                fontWeight: FontWeight.w900,
                color: Colors.black87,
              ),
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
            ),
          ],
        ),
      ),
    );
  }
}
