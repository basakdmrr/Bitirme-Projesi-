import 'dart:async';
import 'dart:io';

import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:record/record.dart';
import 'package:sqflite/sqflite.dart';

void main() => runApp(const BreathAIApp());

/* ===========================
   ✅  API AYARI (BURAYI DÜZENLE)
   ===========================

   - Android Emulator -> PC'deki API:  http://10.0.2.2:8000
   - Gerçek telefon  -> PC aynı Wi-Fi: http://192.168.1.xx:8000 (PC IP)
*/
const String kBaseUrl = "http://10.0.2.2:8000";
const String kPredictPath = "/predict"; // FastAPI: POST /predict

class BreathAIApp extends StatelessWidget {
  const BreathAIApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Breath AI',
      theme: ThemeData(useMaterial3: true),
      home: const SplashScreen(),
    );
  }
}

/* ===========================
   DATA: SQLite (History)
   =========================== */

class AnalysisRecord {
  final int? id;
  final String createdAtIso;
  final String label;
  final double confidence;
  final String audioPath;

  AnalysisRecord({
    this.id,
    required this.createdAtIso,
    required this.label,
    required this.confidence,
    required this.audioPath,
  });

  Map<String, Object?> toMap() => {
        "id": id,
        "createdAtIso": createdAtIso,
        "label": label,
        "confidence": confidence,
        "audioPath": audioPath,
      };

  static AnalysisRecord fromMap(Map<String, Object?> m) => AnalysisRecord(
        id: (m["id"] as int?) ?? 0,
        createdAtIso: (m["createdAtIso"] as String?) ?? "",
        label: (m["label"] as String?) ?? "Bilinmiyor",
        confidence: ((m["confidence"] as num?) ?? 0).toDouble(),
        audioPath: (m["audioPath"] as String?) ?? "",
      );
}

class DbHelper {
  static final DbHelper instance = DbHelper._();
  DbHelper._();

  Database? _db;

  Future<Database> get db async {
    _db ??= await _open();
    return _db!;
  }

  Future<Database> _open() async {
    final base = await getDatabasesPath();
    final path = p.join(base, "breath_ai.db");

    return openDatabase(
      path,
      version: 1,
      onCreate: (d, v) async {
        await d.execute("""
          CREATE TABLE analysis_records(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            createdAtIso TEXT NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            audioPath TEXT NOT NULL
          );
        """);
      },
    );
  }

  Future<int> insertRecord(AnalysisRecord r) async {
    final d = await db;
    return d.insert("analysis_records", r.toMap());
  }

  Future<List<AnalysisRecord>> getRecords() async {
    final d = await db;
    final rows = await d.query("analysis_records", orderBy: "id DESC");
    return rows.map(AnalysisRecord.fromMap).toList();
  }

  Future<void> clearAll() async {
    final d = await db;
    await d.delete("analysis_records");
  }
}

/* ===========================
   API SERVICE (FastAPI /predict)
   =========================== */

class ApiService {
  final Dio _dio = Dio(
    BaseOptions(
      baseUrl: kBaseUrl,
      connectTimeout: const Duration(seconds: 20),
      receiveTimeout: const Duration(seconds: 90),
    ),
  );

  Future<PredictionResult> predictWav(File wavFile) async {
    final fileName = p.basename(wavFile.path);

    final formData = FormData.fromMap({
      "file": await MultipartFile.fromFile(wavFile.path, filename: fileName),
    });

    final res = await _dio.post(kPredictPath, data: formData);

    final data = res.data;
    if (data is! Map) {
      throw Exception("API JSON formatı bekleniyor. Gelen: ${res.data}");
    }

    final label = (data["prediction"] ?? data["label"] ?? "Bilinmiyor").toString();
    final confRaw = (data["confidence"] ?? data["score"] ?? 0.0);

    final confidence = confRaw is num ? confRaw.toDouble() : (double.tryParse(confRaw.toString()) ?? 0.0);

    return PredictionResult(label: label, confidence: confidence, raw: Map<String, dynamic>.from(data));
  }
}

class PredictionResult {
  final String label;
  final double confidence;
  final Map<String, dynamic> raw;

  PredictionResult({required this.label, required this.confidence, required this.raw});
}

/* ===========================
   UI COMMON BACKGROUND
   =========================== */

class _Bg extends StatelessWidget {
  final Widget child;
  final bool light;
  const _Bg({required this.child, this.light = false});

  @override
  Widget build(BuildContext context) {
    final colors = light
        ? const [Color(0xFFE9F7FF), Color(0xFFBFEFFF), Color(0xFFA7F3D0)]
        : const [Color(0xFF0B4A7A), Color(0xFF0EA5E9), Color(0xFF14B8A6)];

    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(begin: Alignment.topLeft, end: Alignment.bottomRight, colors: colors),
      ),
      child: child,
    );
  }
}

/* ===========================
   1) SPLASH
   =========================== */

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});
  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();
    Future.delayed(const Duration(seconds: 2), () {
      if (!mounted) return;
      Navigator.pushReplacement(context, MaterialPageRoute(builder: (_) => const RoleSelectScreen()));
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _Bg(
        child: SafeArea(
          child: Column(
            children: [
              const Spacer(flex: 3),
              Container(
                width: 110,
                height: 110,
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.14),
                  borderRadius: BorderRadius.circular(28),
                  border: Border.all(color: Colors.white.withOpacity(0.22)),
                ),
                child: const Icon(Icons.monitor_heart_rounded, color: Colors.white, size: 54),
              ),
              const SizedBox(height: 18),
              const Text("Breath AI", style: TextStyle(fontSize: 34, fontWeight: FontWeight.w900, color: Colors.white)),
              const SizedBox(height: 10),
              Text(
                "Yapay Zeka Destekli\nSolunum Analizi",
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600, height: 1.2, color: Colors.white.withOpacity(0.85)),
              ),
              const Spacer(flex: 4),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: const [
                  _Dot(active: true),
                  SizedBox(width: 7),
                  _Dot(active: false),
                  SizedBox(width: 7),
                  _Dot(active: false),
                ],
              ),
              const SizedBox(height: 18),
            ],
          ),
        ),
      ),
    );
  }
}

class _Dot extends StatelessWidget {
  final bool active;
  const _Dot({required this.active});

  @override
  Widget build(BuildContext context) {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 250),
      width: active ? 12 : 8,
      height: 8,
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(active ? 0.95 : 0.45),
        borderRadius: BorderRadius.circular(20),
      ),
    );
  }
}

/* ===========================
   2) ROLE SELECT
   =========================== */

class RoleSelectScreen extends StatelessWidget {
  const RoleSelectScreen({super.key});

  void _goHome(BuildContext context, String role) {
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (_) => HomeScreen(role: role, userName: "Kullanıcı")),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Giriş Seçimi", style: TextStyle(fontWeight: FontWeight.w900))),
      body: _Bg(
        light: true,
        child: Center(
          child: Container(
            width: 340,
            padding: const EdgeInsets.all(18),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.90),
              borderRadius: BorderRadius.circular(22),
              border: Border.all(color: Colors.white.withOpacity(0.7)),
              boxShadow: [BoxShadow(blurRadius: 18, offset: const Offset(0, 10), color: Colors.black.withOpacity(0.10))],
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Text("Devam etmek için seçin", style: TextStyle(fontSize: 18, fontWeight: FontWeight.w900)),
                const SizedBox(height: 14),
                _RoleButton(
                  text: "Kullanıcı Girişi (Uzman/Doktor)",
                  icon: Icons.medical_services_rounded,
                  color: const Color(0xFF10B981),
                  onTap: () => _goHome(context, "Doktor"),
                ),
                const SizedBox(height: 12),
                _RoleButton(
                  text: "Hasta Girişi",
                  icon: Icons.person_rounded,
                  color: const Color(0xFF14B8A6),
                  onTap: () => _goHome(context, "Hasta"),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _RoleButton extends StatelessWidget {
  final String text;
  final IconData icon;
  final Color color;
  final VoidCallback onTap;

  const _RoleButton({required this.text, required this.icon, required this.color, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return InkWell(
      borderRadius: BorderRadius.circular(16),
      onTap: onTap,
      child: Container(
        height: 62,
        padding: const EdgeInsets.symmetric(horizontal: 14),
        decoration: BoxDecoration(
          gradient: LinearGradient(colors: [color, color.withOpacity(0.75)]),
          borderRadius: BorderRadius.circular(16),
          boxShadow: [BoxShadow(blurRadius: 14, offset: const Offset(0, 8), color: color.withOpacity(0.22))],
        ),
        child: Row(
          children: [
            CircleAvatar(backgroundColor: Colors.white.withOpacity(0.22), child: Icon(icon, color: Colors.white)),
            const SizedBox(width: 12),
            Expanded(child: Text(text, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w800, color: Colors.white))),
            const Icon(Icons.chevron_right_rounded, color: Colors.white, size: 28),
          ],
        ),
      ),
    );
  }
}

/* ===========================
   3) HOME
   =========================== */

class HomeScreen extends StatefulWidget {
  final String role;
  final String userName;
  const HomeScreen({super.key, required this.role, required this.userName});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

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

  Future<void> _goHistory() async {
    await Navigator.push(context, MaterialPageRoute(builder: (_) => const HistoryScreen()));
    await _loadLast();
  }

  Future<void> _goRecord() async {
    await Navigator.push(context, MaterialPageRoute(builder: (_) => const RecordScreen()));
    await _loadLast();
  }

  @override
  Widget build(BuildContext context) {
    final letter = widget.role.isNotEmpty ? widget.role.characters.first : "U";

    return Scaffold(
      body: _Bg(
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


/* ===========================
   4) RECORD (REAL WAV RECORD)
   =========================== */

class RecordScreen extends StatefulWidget {
  const RecordScreen({super.key});
  @override
  State<RecordScreen> createState() => _RecordScreenState();
}

class _RecordScreenState extends State<RecordScreen> {
  final AudioRecorder _recorder = AudioRecorder();
  bool recording = false;

  static const int totalSeconds = 10;
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
    return p.join(dir.path, "breath_${DateTime.now().millisecondsSinceEpoch}.wav");
  }

  Future<void> _start() async {
    try {
      setState(() => error = null);

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

      audioPath = path;
      recording = true;
      sec = 0;

      t?.cancel();
      t = Timer.periodic(const Duration(seconds: 1), (_) async {
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
      await _recorder.stop();
      recording = false;

      if (!mounted) return;
      if (audioPath == null) {
        setState(() => error = "Ses dosyası oluşmadı.");
        return;
      }

      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (_) => AnalyzingScreen(audioPath: audioPath!)),
      );
    } catch (e) {
      setState(() => error = "Kayıt durdurulamadı: $e");
    }
  }

  @override
  Widget build(BuildContext context) {
    final progress = (sec / totalSeconds).clamp(0.0, 1.0);

    return Scaffold(
      body: _Bg(
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
                    const Text("Kayıt", style: TextStyle(color: Colors.white, fontWeight: FontWeight.w900)),
                    const Spacer(),
                    const SizedBox(width: 48),
                  ],
                ),
                const Spacer(),
                Icon(recording ? Icons.graphic_eq_rounded : Icons.mic_none_rounded, size: 120, color: Colors.white),
                const SizedBox(height: 14),
                Text(
                  recording ? "Lütfen nefes verin...\nKayıt alınıyor." : "Hazır olduğunda kaydı başlat.",
                  textAlign: TextAlign.center,
                  style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w700, height: 1.2),
                ),
                const SizedBox(height: 14),
                Text(
                  "$sec / $totalSeconds sn",
                  style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w900, fontSize: 18),
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
                  Text(error!, textAlign: TextAlign.center, style: const TextStyle(color: Colors.yellow, fontWeight: FontWeight.w800)),
                const Spacer(),
                SizedBox(
                  width: double.infinity,
                  height: 54,
                  child: ElevatedButton(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.white,
                      foregroundColor: const Color(0xFF0B4A7A),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                    ),
                    onPressed: recording ? _stopAndGoAnalyze : _start,
                    child: Text(recording ? "Kaydı Bitir & Analiz Et" : "Kaydı Başlat", style: const TextStyle(fontWeight: FontWeight.w900)),
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

/* ===========================
   5) ANALYZING (CALL API)
   =========================== */

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
      final api = ApiService();
      final res = await api.predictWav(File(widget.audioPath));

      final rec = AnalysisRecord(
        createdAtIso: DateTime.now().toIso8601String(),
        label: res.label,
        confidence: res.confidence,
        audioPath: widget.audioPath,
      );
      await DbHelper.instance.insertRecord(rec);

      if (!mounted) return;
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) => ResultScreen(
            result: res.label,
            confidence: res.confidence,
            audioPath: widget.audioPath,
            createdAtIso: rec.createdAtIso,
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
      body: _Bg(
        child: SafeArea(
          child: Center(
            child: Padding(
              padding: const EdgeInsets.all(18),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.memory_rounded, color: Colors.white, size: 60),
                  const SizedBox(height: 12),
                  const Text("Analiz Ediliyor...", style: TextStyle(color: Colors.white, fontWeight: FontWeight.w900, fontSize: 18)),
                  const SizedBox(height: 10),
                  if (error == null)
                    SizedBox(
                      width: 240,
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(999),
                        child: LinearProgressIndicator(
                          minHeight: 10,
                          backgroundColor: Colors.white.withOpacity(0.25),
                          valueColor: const AlwaysStoppedAnimation<Color>(Colors.white),
                        ),
                      ),
                    )
                  else ...[
                    const SizedBox(height: 12),
                    Text("API Hatası:\n$error", textAlign: TextAlign.center, style: const TextStyle(color: Colors.yellow, fontWeight: FontWeight.w800)),
                    const SizedBox(height: 12),
                    SizedBox(
                      width: double.infinity,
                      height: 50,
                      child: ElevatedButton(
                        style: ElevatedButton.styleFrom(backgroundColor: Colors.white, foregroundColor: const Color(0xFF0B4A7A)),
                        onPressed: () => Navigator.pop(context),
                        child: const Text("Geri Dön", style: TextStyle(fontWeight: FontWeight.w900)),
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

/* ===========================
   6) RESULT
   =========================== */

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
    final ok = result.toLowerCase().contains("healthy") || result.toLowerCase().contains("sağ");

    return Scaffold(
      body: _Bg(
        light: true,
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(18),
            child: Column(
              children: [
                Row(
                  children: [
                    IconButton(onPressed: () => Navigator.pop(context), icon: const Icon(Icons.arrow_back_rounded)),
                    const Spacer(),
                    const Text("Sonuç", style: TextStyle(fontWeight: FontWeight.w900, fontSize: 18)),
                    const Spacer(),
                    IconButton(
                      onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const HistoryScreen())),
                      icon: const Icon(Icons.history_rounded),
                    ),
                  ],
                ),
                const Spacer(),
                CircleAvatar(
                  radius: 46,
                  backgroundColor: ok ? const Color(0xFF10B981) : const Color(0xFFF59E0B),
                  child: Icon(ok ? Icons.check_rounded : Icons.warning_rounded, color: Colors.white, size: 56),
                ),
                const SizedBox(height: 14),
                Text("Tahmin: $result", style: const TextStyle(fontSize: 22, fontWeight: FontWeight.w900)),
                const SizedBox(height: 6),
                Text("Güven Skoru: %$percent", style: TextStyle(color: Colors.black.withOpacity(0.65), fontWeight: FontWeight.w700)),
                const SizedBox(height: 10),
                Text("Tarih: ${createdAtIso.substring(0, 19).replaceAll('T', ' ')}", style: TextStyle(color: Colors.black.withOpacity(0.55), fontWeight: FontWeight.w600)),
                const SizedBox(height: 14),
                SizedBox(
                  width: double.infinity,
                  height: 54,
                  child: ElevatedButton(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF0EA5E9),
                      foregroundColor: Colors.white,
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                    ),
                    onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const HistoryScreen())),
                    child: const Text("Analiz Geçmişi", style: TextStyle(fontWeight: FontWeight.w900)),
                  ),
                ),
                const SizedBox(height: 10),
                SizedBox(
                  width: double.infinity,
                  height: 54,
                  child: OutlinedButton(
                    style: OutlinedButton.styleFrom(shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16))),
                    onPressed: () {
                      Navigator.pushAndRemoveUntil(
                        context,
                        MaterialPageRoute(builder: (_) => const RoleSelectScreen()),
                        (_) => false,
                      );
                    },
                    child: const Text("Başa Dön", style: TextStyle(fontWeight: FontWeight.w900)),
                  ),
                ),
                const SizedBox(height: 10),
                Text(
                  "Not: Bu bir ön değerlendirmedir, kesin tanı için hekime başvurun.",
                  textAlign: TextAlign.center,
                  style: TextStyle(color: Colors.black.withOpacity(0.55), fontWeight: FontWeight.w600),
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

/* ===========================
   7) HISTORY (FROM SQLITE)
   =========================== */

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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Analiz Geçmişi", style: TextStyle(fontWeight: FontWeight.w900)),
        actions: [
          IconButton(
            onPressed: () async {
              await DbHelper.instance.clearAll();
              await _load();
            },
            icon: const Icon(Icons.delete_forever_rounded),
            tooltip: "Tüm kayıtları sil",
          )
        ],
      ),
      body: _Bg(
        light: true,
        child: loading
            ? const Center(child: CircularProgressIndicator())
            : items.isEmpty
                ? const Center(child: Text("Kayıt yok", style: TextStyle(fontWeight: FontWeight.w900)))
                : ListView.separated(
                    padding: const EdgeInsets.all(18),
                    itemBuilder: (_, i) {
                      final r = items[i];
                      final date = r.createdAtIso.length >= 19 ? r.createdAtIso.substring(0, 19).replaceAll('T', ' ') : r.createdAtIso;
                      final percent = (r.confidence * 100).round();
                      final color = r.label.toLowerCase().contains("healthy") || r.label.toLowerCase().contains("sağ")
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
                          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.90),
                            borderRadius: BorderRadius.circular(16),
                            boxShadow: [BoxShadow(blurRadius: 14, offset: const Offset(0, 8), color: Colors.black.withOpacity(0.08))],
                          ),
                          child: Row(
                            children: [
                              CircleAvatar(
                                backgroundColor: color.withOpacity(0.15),
                                child: Icon(Icons.monitor_heart_rounded, color: color),
                              ),
                              const SizedBox(width: 12),
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(date, style: const TextStyle(fontWeight: FontWeight.w900)),
                                    const SizedBox(height: 2),
                                    Text("${r.label}  •  %$percent", style: TextStyle(color: color, fontWeight: FontWeight.w900)),
                                  ],
                                ),
                              ),
                              const Icon(Icons.chevron_right_rounded),
                            ],
                          ),
                        ),
                      );
                    },
                    separatorBuilder: (_, __) => const SizedBox(height: 12),
                    itemCount: items.length,
                  ),
      ),
    );
  }
}
