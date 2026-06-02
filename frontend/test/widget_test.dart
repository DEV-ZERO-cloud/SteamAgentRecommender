import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:steam_agent_frontend/theme/app_theme.dart';
import 'package:steam_agent_frontend/screens/home_screen.dart';

void main() {
  testWidgets('App renders header and input', (WidgetTester tester) async {
    await tester.pumpWidget(
      ProviderScope(
        child: MaterialApp(
          theme: AppTheme.darkTheme,
          home: const HomeScreen(),
        ),
      ),
    );

    expect(find.text('SteamAgent'), findsOneWidget);
    expect(find.text('Recomendador Inteligente de RPGs'), findsOneWidget);
    expect(find.text('Buscar'), findsOneWidget);
  });
}
