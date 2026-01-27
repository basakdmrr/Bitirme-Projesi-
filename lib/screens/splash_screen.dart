import 'package:flutter/material.dart';
import 'role_select_screen.dart';
import '../widgets/bg.dart';
/* ===========================
   UI COMMON BACKGROUND
   =========================== */

class Bg extends StatelessWidget {
  final Widget child;
  final bool light;
  const Bg({required this.child, this.light = false});

  @override
  Widget build(BuildContext context) {
    final colors = light
        ? const [
            Color(0xFFE9F7FF),
            Color(0xFFBFEFFF),
            Color(0xFFA7F3D0),
          ]
        : const [
            Color(0xFF0B4A7A),
            Color(0xFF0EA5E9),
            Color(0xFF14B8A6),
          ];

    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: colors,
        ),
      ),
      child: child,
    );
  }
}

/* ===========================
   SPLASH
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
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) => const RoleSelectScreen(),
        ),
      );
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
                  border: Border.all(
                    color: Colors.white.withOpacity(0.22),
                  ),
                ),
                child: const Icon(
                  Icons.monitor_heart_rounded,
                  color: Colors.white,
                  size: 54,
                ),
              ),
              const SizedBox(height: 18),
              const Text(
                "Breath AI",
                style: TextStyle(
                  fontSize: 34,
                  fontWeight: FontWeight.w900,
                  color: Colors.white,
                ),
              ),
              const SizedBox(height: 10),
              Text(
                "Yapay Zeka Destekli\nSolunum Analizi",
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                  height: 1.2,
                  color: Colors.white.withOpacity(0.85),
                ),
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

/* ===========================
   DOT
   =========================== */

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
