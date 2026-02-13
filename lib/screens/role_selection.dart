import 'package:flutter/material.dart';
import '../widgets/role_button.dart';
import 'login_page.dart';

class RoleSelectScreen extends StatelessWidget {
  const RoleSelectScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          "Giriş Seçimi",
          style: TextStyle(fontWeight: FontWeight.w900),
        ),
      ),
      body: Center(
        child: Container(
          width: 340,
          padding: const EdgeInsets.all(18),
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.90),
            borderRadius: BorderRadius.circular(22),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text(
                "Devam etmek için seçin",
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.w900),
              ),
              const SizedBox(height: 14),

              RoleButton(
                text: "Doktor Girişi",
                icon: Icons.medical_services_rounded,
                color: const Color(0xFF10B981),
                onTap: () {
                  // 🔕 Şimdilik hiçbir şey yapmıyor
                },
              ),
            ],
          ),
        ),
      ),
    );
  }
}
