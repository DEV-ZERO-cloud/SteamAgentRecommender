import 'package:dio/dio.dart';
import '../models/profile.dart';
import '../models/recommendation.dart';
import '../models/game.dart';

class SteamApiService {
  final Dio _dio;

  SteamApiService(String baseUrl)
      : _dio = Dio(BaseOptions(
          baseUrl: baseUrl,
          connectTimeout: const Duration(seconds: 30),
          receiveTimeout: const Duration(seconds: 60),
          headers: {'Content-Type': 'application/json'},
        ));

  Future<SteamProfile> getProfile(String username) async {
    final response =
        await _dio.post('/profile', data: {'username': username});
    return SteamProfile.fromJson(response.data);
  }

  Future<List<Recommendation>> getRecommendations({
    required List<String> query,
    List<String>? dislikedTags,
    int topK = 10,
    bool isPrice = false,
    double minPrice = 0.0,
    double maxPrice = 999999.0,
    bool isPositiveRate = false,
    double minPositiveRate = 0.0,
    bool isDate = false,
    int minYear = 0,
    int maxYear = 9999,
    bool isRecommendations = false,
    double minRecommendations = 0.0,
  }) async {
    final response = await _dio.post('/recommend', data: {
      'query': query,
      'disliked_tags': dislikedTags,
      'top_k': topK,
      'isPrice': isPrice,
      'MinPrice': minPrice,
      'MaxPrice': maxPrice,
      'isPositiveRate': isPositiveRate,
      'MinPositiveRate': minPositiveRate,
      'isDate': isDate,
      'MinYear': minYear,
      'MaxYear': maxYear,
      'isRecommendations': isRecommendations,
      'MinRecommendations': minRecommendations,
    });
    final results = (response.data['results'] as List<dynamic>)
        .map((e) => Recommendation.fromJson(e as Map<String, dynamic>))
        .toList();
    return results;
  }

  Future<Game> getGameDetails(int appId) async {
    final response = await _dio.get('/games/$appId');
    return Game.fromJson(response.data);
  }
}
