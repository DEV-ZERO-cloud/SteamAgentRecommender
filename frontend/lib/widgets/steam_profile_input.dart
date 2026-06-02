import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import '../providers/recommendation_provider.dart';
import '../theme/app_theme.dart';

class SteamProfileInput extends ConsumerStatefulWidget {
  const SteamProfileInput({super.key});

  @override
  ConsumerState<SteamProfileInput> createState() => _SteamProfileInputState();
}

class _SteamProfileInputState extends ConsumerState<SteamProfileInput> {
  final _controller = TextEditingController();

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _submit() {
    final text = _controller.text.trim();
    if (text.isNotEmpty) {
      ref.read(recommendationProvider.notifier).searchBySteamId(text);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
      child: Row(
        children: [
          Expanded(
            child: TextField(
              controller: _controller,
              onSubmitted: (_) => _submit(),
              decoration: const InputDecoration(
                hintText: 'Ingresa tu SteamID o Custom URL...',
                prefixIcon: Icon(Icons.person_search,
                    color: AppTheme.textSecondary),
              ),
              style:
                  GoogleFonts.inter(color: Colors.white, fontSize: 16),
            ),
          ),
          const SizedBox(width: 12),
          SizedBox(
            height: 52,
            child: ElevatedButton(
              onPressed: _submit,
              child: const Text('Buscar'),
            ),
          ),
        ],
      ),
    );
  }
}
