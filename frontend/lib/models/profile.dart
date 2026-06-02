class SteamProfile {
  final String userId;
  final String name;
  final List<int> playedAppIds;
  final List<String> preferredTags;
  final List<int> amountPlaying;

  SteamProfile({
    required this.userId,
    required this.name,
    required this.playedAppIds,
    required this.preferredTags,
    required this.amountPlaying,
  });

  factory SteamProfile.fromJson(Map<String, dynamic> json) {
    return SteamProfile(
      userId: json['user_id'] as String,
      name: json['name'] as String,
      playedAppIds: (json['played_app_ids'] as List<dynamic>)
          .map((e) => e as int)
          .toList(),
      preferredTags: (json['preferred_tags'] as List<dynamic>)
          .map((e) => e as String)
          .toList(),
      amountPlaying: (json['amount_playing'] as List<dynamic>)
          .map((e) => e as int)
          .toList(),
    );
  }
}
