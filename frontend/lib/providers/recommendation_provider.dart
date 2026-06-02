import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/profile.dart';
import '../models/recommendation.dart';
import '../models/game.dart';
import '../services/steam_api_service.dart';

enum AppState { idle, loadingProfile, profileLoaded, loadingGames, loaded, error }

class FilterParams {
  final double maxPrice;
  final double minRating;
  final int? minYear;
  final int? maxYear;
  final double minRecommendations;

  const FilterParams({
    this.maxPrice = 999999.0,
    this.minRating = 0.0,
    this.minYear,
    this.maxYear,
    this.minRecommendations = 0.0,
  });

  FilterParams copyWith({
    double? maxPrice,
    double? minRating,
    int? minYear,
    int? maxYear,
    double? minRecommendations,
  }) {
    return FilterParams(
      maxPrice: maxPrice ?? this.maxPrice,
      minRating: minRating ?? this.minRating,
      minYear: minYear ?? this.minYear,
      maxYear: maxYear ?? this.maxYear,
      minRecommendations: minRecommendations ?? this.minRecommendations,
    );
  }
}

class RecommendationState {
  final AppState state;
  final String? error;
  final SteamProfile? profile;
  final List<Recommendation> recommendations;
  final Map<int, Game> gameDetails;
  final FilterParams filters;

  RecommendationState({
    this.state = AppState.idle,
    this.error,
    this.profile,
    this.recommendations = const [],
    this.gameDetails = const {},
    this.filters = const FilterParams(),
  });

  RecommendationState copyWith({
    AppState? state,
    String? error,
    SteamProfile? profile,
    List<Recommendation>? recommendations,
    Map<int, Game>? gameDetails,
    FilterParams? filters,
  }) {
    return RecommendationState(
      state: state ?? this.state,
      error: error,
      profile: profile ?? this.profile,
      recommendations: recommendations ?? this.recommendations,
      gameDetails: gameDetails ?? this.gameDetails,
      filters: filters ?? this.filters,
    );
  }

  List<Recommendation> get filteredRecommendations {
    return recommendations.where((rec) {
      final details = gameDetails[rec.gameScore.appId];
      if (details == null) return true;
      if (details.price > filters.maxPrice) return false;
      if (details.rating < filters.minRating) return false;
      if (filters.minYear != null &&
          (details.releaseYear ?? 0) < filters.minYear!) {
        return false;
      }
      if (filters.maxYear != null &&
          (details.releaseYear ?? 9999) > filters.maxYear!) {
        return false;
      }
      if (details.positiveReviews < filters.minRecommendations) return false;
      return true;
    }).toList();
  }
}

class RecommendationNotifier extends StateNotifier<RecommendationState> {
  final SteamApiService _api;

  RecommendationNotifier(this._api) : super(RecommendationState());

  Future<void> searchBySteamId(String username) async {
    state = state.copyWith(state: AppState.loadingProfile, error: null);
    try {
      final profile = await _api.getProfile(username);
      state = state.copyWith(
        state: AppState.profileLoaded,
        profile: profile,
      );
      await loadRecommendations(profile.preferredTags);
    } catch (e) {
      state = state.copyWith(
        state: AppState.error,
        error: 'Error al cargar perfil: ${e.toString()}',
      );
    }
  }

  Future<void> loadRecommendations(List<String> tags) async {
    state = state.copyWith(state: AppState.loadingGames);
    try {
      final recommendations =
          await _api.getRecommendations(query: tags, topK: 20);
      state = state.copyWith(
        state: AppState.loaded,
        recommendations: recommendations,
      );
      await _fetchGameDetails(recommendations);
    } catch (e) {
      state = state.copyWith(
        state: AppState.error,
        error: 'Error al obtener recomendaciones: ${e.toString()}',
      );
    }
  }

  Future<void> _fetchGameDetails(List<Recommendation> recs) async {
    final details = <int, Game>{};
    try {
      final futures =
          recs.map((r) => _api.getGameDetails(r.gameScore.appId));
      final results = await Future.wait(futures);
      for (final game in results) {
        details[game.appId] = game;
      }
    } catch (_) {}
    state = state.copyWith(gameDetails: details);
  }

  void updateFilters(FilterParams filters) {
    state = state.copyWith(filters: filters);
  }
}

const String defaultApiUrl = String.fromEnvironment(
  'API_URL',
  defaultValue: 'http://localhost:8000',
);

final steamApiServiceProvider = Provider<SteamApiService>((ref) {
  return SteamApiService(defaultApiUrl);
});

final recommendationProvider =
    StateNotifierProvider<RecommendationNotifier, RecommendationState>((ref) {
  final api = ref.watch(steamApiServiceProvider);
  return RecommendationNotifier(api);
});
