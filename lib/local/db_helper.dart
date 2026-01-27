import 'package:path/path.dart' as p;
import 'package:sqflite/sqflite.dart';
import 'analysis_record.dart';

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
      onCreate: (d, _) async {
        await d.execute('''
          CREATE TABLE analysis_records(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            createdAtIso TEXT NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            audioPath TEXT NOT NULL
          )
        ''');
      },
    );
  }

  Future<int> insertRecord(AnalysisRecord r) async {
    final d = await db;
    return d.insert("analysis_records", r.toMap());
  }

  Future<List<AnalysisRecord>> getRecords() async {
    final d = await db;
    final rows = await d.query(
      "analysis_records",
      orderBy: "id DESC",
    );
    return rows.map(AnalysisRecord.fromMap).toList();
  }

  Future<void> clearAll() async {
    final d = await db;
    await d.delete("analysis_records");
  }
}
