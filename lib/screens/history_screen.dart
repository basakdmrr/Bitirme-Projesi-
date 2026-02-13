import 'package:flutter/material.dart';
import '../services/history_service.dart';
import '../db/db_helper.dart';
import 'result_screen.dart';
class HistoryScreen extends StatefulWidget {
  final String token;
  
  const HistoryScreen({super.key, required this.token});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  List<AnalysisRecord> _records = [];
  bool _isLoading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadRecords();
  }

  Future<void> _loadRecords() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      // Önce lokal veritabanından çek
      final localRecords = await DbHelper.instance.getRecords();
      
      setState(() {
        _records = localRecords;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Geçmiş Analizler'),
        backgroundColor: const Color(0xFF0B4A7A),
        foregroundColor: Colors.white,
      ),
      body: Container(
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
        child: _isLoading
            ? const Center(
                child: CircularProgressIndicator(color: Colors.white),
              )
            : _error != null
                ? Center(
                    child: Text(
                      'Hata: $_error',
                      style: const TextStyle(color: Colors.white),
                    ),
                  )
                : _records.isEmpty
                    ? const Center(
                        child: Text(
                          'Henüz kayıt yok',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 18,
                          ),
                        ),
                      )
                    : RefreshIndicator(
                        onRefresh: _loadRecords,
                        child: ListView.builder(
                          padding: const EdgeInsets.all(16),
                          itemCount: _records.length,
                          itemBuilder: (context, index) {
                            final record = _records[index];
                            return _RecordCard(record: record);
                          },
                        ),
                      ),
      ),
    );
  }
}

class _RecordCard extends StatelessWidget {
  final AnalysisRecord record;
  
  const _RecordCard({required this.record});

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: ListTile(
        leading: CircleAvatar(
          backgroundColor: const Color(0xFF0B4A7A),
          child: Text(
            '${(record.confidence * 100).toInt()}%',
            style: const TextStyle(
              color: Colors.white,
              fontSize: 12,
              fontWeight: FontWeight.bold,
            ),
          ),
        ),
        title: Text(
          record.label,
          style: const TextStyle(fontWeight: FontWeight.bold),
        ),
        subtitle: Text(
          _formatDate(record.createdAtIso),
          style: TextStyle(color: Colors.grey[600]),
        ),
        trailing: const Icon(Icons.chevron_right),
        onTap: () {
          // ResultScreen'e git
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (_) => ResultScreen(
                result: record.label,
                confidence: record.confidence,
                audioPath: record.audioPath,
                createdAtIso: record.createdAtIso,
                fromHistory: true,
              ),
            ),
          );
        },
      ),
    );
  }

  String _formatDate(String isoDate) {
    try {
      final date = DateTime.parse(isoDate);
      return '${date.day}.${date.month}.${date.year} ${date.hour}:${date.minute.toString().padLeft(2, '0')}';
    } catch (e) {
      return isoDate;
    }
  }
}
