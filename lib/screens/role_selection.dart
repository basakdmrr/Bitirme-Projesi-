import '../widgets/bg.dart' ;
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
      body: Bg(
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
