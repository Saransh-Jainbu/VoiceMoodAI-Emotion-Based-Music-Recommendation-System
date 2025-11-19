import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
import os
import json
import requests
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use environment variables directly

class SpotifyRecommender:
    def __init__(self, client_id: str = None, client_secret: str = None, use_oauth: bool = False, redirect_uri: str = None):
        """
        Initialize Spotify recommender with client credentials or OAuth

        Args:
            client_id: Spotify API client ID
            client_secret: Spotify API client secret
            use_oauth: Whether to use OAuth for user authentication
            redirect_uri: Redirect URI for OAuth (required if use_oauth=True)
        """
        self.client_id = client_id or os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('SPOTIFY_CLIENT_SECRET')
        self.use_oauth = use_oauth
        self.redirect_uri = redirect_uri or os.getenv('SPOTIFY_REDIRECT_URI', 'http://127.0.0.1:8501')

        if not self.client_id or not self.client_secret:
            print("⚠️  Spotify credentials not found!")
            print("   Please set environment variables:")
            print("   export SPOTIFY_CLIENT_ID='your_client_id'")
            print("   export SPOTIFY_CLIENT_SECRET='your_client_secret'")
            if use_oauth:
                print("   export SPOTIFY_REDIRECT_URI='your_redirect_uri'")
            print("   Or get them from: https://developer.spotify.com/dashboard")
            self.sp = None
            return

        try:
            if use_oauth:
                # Use OAuth for user authentication (allows playlist creation, user data access)
                oauth_manager = SpotifyOAuth(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    redirect_uri=self.redirect_uri,
                    scope='playlist-modify-public playlist-modify-private user-library-read'
                )
                self.sp = spotipy.Spotify(auth_manager=oauth_manager)
                print("✅ Spotify OAuth connected successfully!")
            else:
                # Use Client Credentials for server-side recommendations only
                client_credentials_manager = SpotifyClientCredentials(
                    client_id=self.client_id,
                    client_secret=self.client_secret
                )
                self.sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
                print("✅ Spotify API connected successfully!")
        except Exception as e:
            print(f"❌ Failed to connect to Spotify API: {e}")
            self.sp = None

    def emotion_to_spotify_features(self, emotion: str) -> Dict[str, float]:
        """
        Map emotions to Spotify audio features for recommendations

        Args:
            emotion: Detected emotion (happy, sad, angry, etc.)

        Returns:
            Dictionary of Spotify audio features
        """
        # Base features for different emotions
        emotion_features = {
            'happy': {
                'valence': 0.8,      # High positivity
                'energy': 0.8,       # High energy
                'danceability': 0.7, # Danceable
                'tempo': 120,        # Upbeat tempo
                'mode': 1,           # Major key
                'acousticness': 0.2, # Not too acoustic
                'instrumentalness': 0.1, # Some vocals preferred
                'liveness': 0.2      # Studio recordings
            },
            'sad': {
                'valence': 0.2,      # Low positivity
                'energy': 0.3,       # Low energy
                'danceability': 0.3, # Not very danceable
                'tempo': 80,         # Slow tempo
                'mode': 0,           # Minor key
                'acousticness': 0.6, # More acoustic
                'instrumentalness': 0.3, # Can be instrumental
                'liveness': 0.1      # Studio recordings
            },
            'angry': {
                'valence': 0.3,      # Low positivity
                'energy': 0.9,       # Very high energy
                'danceability': 0.6, # Moderately danceable
                'tempo': 140,        # Fast tempo
                'mode': 0,           # Minor key
                'acousticness': 0.1, # Electronic/rock
                'instrumentalness': 0.2, # Some vocals
                'liveness': 0.3      # Can be live
            },
            'calm': {
                'valence': 0.5,      # Neutral positivity
                'energy': 0.3,       # Low energy
                'danceability': 0.4, # Not very danceable
                'tempo': 90,         # Moderate tempo
                'mode': 1,           # Major key
                'acousticness': 0.7, # Very acoustic
                'instrumentalness': 0.4, # Can be instrumental
                'liveness': 0.1      # Studio recordings
            },
            'fear': {
                'valence': 0.2,      # Low positivity
                'energy': 0.4,       # Moderate energy
                'danceability': 0.3, # Not danceable
                'tempo': 100,        # Moderate tempo
                'mode': 0,           # Minor key
                'acousticness': 0.5, # Mixed
                'instrumentalness': 0.6, # More instrumental
                'liveness': 0.2      # Studio
            },
            'disgust': {
                'valence': 0.3,      # Low positivity
                'energy': 0.5,       # Moderate energy
                'danceability': 0.4, # Not very danceable
                'tempo': 110,        # Moderate tempo
                'mode': 0,           # Minor key
                'acousticness': 0.4, # Mixed
                'instrumentalness': 0.3, # Some vocals
                'liveness': 0.2      # Studio
            },
            'neutral': {
                'valence': 0.5,      # Neutral positivity
                'energy': 0.5,       # Moderate energy
                'danceability': 0.5, # Moderately danceable
                'tempo': 110,        # Moderate tempo
                'mode': 0.5,         # Any mode
                'acousticness': 0.4, # Mixed
                'instrumentalness': 0.2, # Some vocals
                'liveness': 0.2      # Studio
            },
            'surprise': {
                'valence': 0.7,      # High positivity
                'energy': 0.7,       # High energy
                'danceability': 0.6, # Danceable
                'tempo': 130,        # Upbeat tempo
                'mode': 1,           # Major key
                'acousticness': 0.3, # Mixed
                'instrumentalness': 0.1, # Vocals preferred
                'liveness': 0.3      # Can be live
            }
        }

        return emotion_features.get(emotion.lower(), emotion_features['neutral'])

    def get_recommendations(self, emotion: str, limit: int = 10) -> List[Dict]:
        """
        Get Spotify recommendations based on detected emotion

        Args:
            emotion: Detected emotion
            limit: Number of recommendations to return

        Returns:
            List of recommended tracks with metadata
        """
        if not self.sp:
            return self._get_fallback_recommendations(emotion, limit)

        try:
            # Try the recommendations API first
            features = self.emotion_to_spotify_features(emotion)

            try:
                # First try with genres
                genres = self._get_genres_for_emotion(emotion)
                results = self.sp.recommendations(
                    seed_genres=genres[:2],  # Limit to 2 genres
                    limit=limit,
                    **{k: v for k, v in features.items() if k not in ['tempo', 'mode']}
                )
            except:
                # Fallback: use audio features only (no seed genres)
                try:
                    results = self.sp.recommendations(
                        limit=limit,
                        **{k: v for k, v in features.items() if k not in ['tempo', 'mode']}
                    )
                except:
                    # Final fallback: use search-based recommendations
                    return self._get_search_based_recommendations(emotion, limit)

            recommendations = []
            for track in results['tracks']:
                # Get audio features for the track
                try:
                    audio_features = self.sp.audio_features(track['id'])[0]
                except:
                    audio_features = None

                recommendation = {
                    'name': track['name'],
                    'artist': track['artists'][0]['name'],
                    'album': track['album']['name'],
                    'spotify_url': track['external_urls']['spotify'],
                    'preview_url': track.get('preview_url'),
                    'image_url': track['album']['images'][0]['url'] if track['album']['images'] else None,
                    'duration_ms': track['duration_ms'],
                    'popularity': track['popularity'],
                    'audio_features': {
                        'danceability': audio_features['danceability'] if audio_features else 0.5,
                        'energy': audio_features['energy'] if audio_features else 0.5,
                        'valence': audio_features['valence'] if audio_features else 0.5,
                        'tempo': audio_features['tempo'] if audio_features else 120,
                        'acousticness': audio_features['acousticness'] if audio_features else 0.5,
                        'instrumentalness': audio_features['instrumentalness'] if audio_features else 0.0
                    } if audio_features else {},
                    'embed_url': f"https://open.spotify.com/embed/track/{track['id']}"
                }
                recommendations.append(recommendation)

            return recommendations

        except Exception as e:
            print(f"❌ Spotify API error: {e}")
            return self._get_fallback_recommendations(emotion, limit)

    def _get_genres_for_emotion(self, emotion: str) -> List[str]:
        """Get Spotify genres that match the emotion (using valid Spotify genre seeds)"""
        genre_mapping = {
            'happy': ['pop', 'dance', 'electro', 'party'],
            'sad': ['sad', 'indie', 'folk', 'acoustic'],
            'angry': ['rock', 'metal', 'punk', 'alternative'],
            'calm': ['ambient', 'classical', 'jazz', 'lo-fi'],
            'fear': ['ambient', 'classical', 'soundtrack', 'electronic'],
            'disgust': ['industrial', 'experimental', 'noise'],
            'neutral': ['indie', 'alternative', 'folk'],
            'surprise': ['electronic', 'pop', 'hip-hop', 'dance']
        }
        return genre_mapping.get(emotion.lower(), ['pop', 'indie'])

    def _get_fallback_recommendations(self, emotion: str, limit: int = 10) -> List[Dict]:
        """Fallback recommendations when Spotify API is not available"""
        fallback_songs = {
            'happy': [
                {'name': 'Happy', 'artist': 'Pharrell Williams', 'spotify_url': 'https://open.spotify.com/track/60nZcImufyMA1MKQY3dcCH'},
                {'name': "Can't Stop the Feeling!", 'artist': 'Justin Timberlake', 'spotify_url': 'https://open.spotify.com/track/1Je1IMUlBXcx1Fz0WE7oPT'},
                {'name': 'Uptown Funk', 'artist': 'Mark Ronson ft. Bruno Mars', 'spotify_url': 'https://open.spotify.com/track/32OlwWuMpZ6b0aNci9wMId'},
                {'name': 'Shake It Off', 'artist': 'Taylor Swift', 'spotify_url': 'https://open.spotify.com/track/5xTtaWoae3wi06K5WfVUUH'},
                {'name': 'Happy Together', 'artist': 'The Turtles', 'spotify_url': 'https://open.spotify.com/track/1JO1xLtVc8mWhIoE3YaCL0'}
            ],
            'sad': [
                {'name': 'Someone Like You', 'artist': 'Adele', 'spotify_url': 'https://open.spotify.com/track/4kflIGiTgMjTkM3hOQoSPe'},
                {'name': 'Hurt', 'artist': 'Johnny Cash', 'spotify_url': 'https://open.spotify.com/track/28cnXtME493VX9NOw9cIUh'},
                {'name': 'Yesterday', 'artist': 'The Beatles', 'spotify_url': 'https://open.spotify.com/track/3BQHpFgAp4l80e1XslIjNI'},
                {'name': 'Tears in Heaven', 'artist': 'Eric Clapton', 'spotify_url': 'https://open.spotify.com/track/2LawezPeJhS3noDZNg9bBG'},
                {'name': 'Nothing Compares 2 U', 'artist': 'Sinéad O\'Connor', 'spotify_url': 'https://open.spotify.com/track/3nvuPQTw2zuFAVuLsC9O7D'}
            ],
            'angry': [
                {'name': 'Breaking the Law', 'artist': 'Judas Priest', 'spotify_url': 'https://open.spotify.com/track/5NIPsWpDjJTFBoPxCUUe1r'},
                {'name': 'Thunderstruck', 'artist': 'AC/DC', 'spotify_url': 'https://open.spotify.com/track/57bgtoPSgt236HzfBOd8kj'},
                {'name': 'Back in Black', 'artist': 'AC/DC', 'spotify_url': 'https://open.spotify.com/track/08mG3Y1vljYA6bvDt4Wqkj'},
                {'name': 'Welcome to the Jungle', 'artist': 'Guns N\' Roses', 'spotify_url': 'https://open.spotify.com/track/0BVRQi9hBFLBL3aNUXhZ3K'},
                {'name': 'Smells Like Teen Spirit', 'artist': 'Nirvana', 'spotify_url': 'https://open.spotify.com/track/4CeeEOM32jQcH3eN9Q2dGj'}
            ],
            'calm': [
                {'name': 'Weightless', 'artist': 'Marconi Union', 'spotify_url': 'https://open.spotify.com/track/1WJzEhDs2ejuIEG0H8aNfg'},
                {'name': 'River Flows in You', 'artist': 'Yiruma', 'spotify_url': 'https://open.spotify.com/track/4BZq1c6SIwL7XK4yGHp7dR'},
                {'name': 'Comptine d\'un autre été', 'artist': 'Yann Tiersen', 'spotify_url': 'https://open.spotify.com/track/0tgBtQ0ISnPQ8rCTvNDt1G'},
                {'name': 'The Night We Met', 'artist': 'Lord Huron', 'spotify_url': 'https://open.spotify.com/track/0QZ5yyl6B6utIWkxeBDxQN'},
                {'name': 'Hurt', 'artist': 'Johnny Cash', 'spotify_url': 'https://open.spotify.com/track/28cnXtME493VX9NOw9cIUh'}
            ]
        }

        songs = fallback_songs.get(emotion.lower(), fallback_songs['happy'])
        return songs[:limit]

    def create_playlist_url(self, track_ids: List[str]) -> str:
        """Create a Spotify playlist URL from track IDs"""
        if not track_ids:
            return ""

        # Create a playlist URL (this would require user authentication for actual creation)
        # For now, return a search URL with the tracks
        track_string = ",".join(track_ids[:5])  # Limit to 5 tracks for URL
        return f"https://open.spotify.com/search/tracks/{track_string}"

    def get_track_embed(self, track_id: str) -> str:
        """Get Spotify embed HTML for a track"""
        return f'<iframe src="https://open.spotify.com/embed/track/{track_id}" width="300" height="380" frameborder="0" allowtransparency="true" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>'

    def get_auth_url(self) -> str:
        """Get the Spotify authorization URL for OAuth flow"""
        if not self.use_oauth:
            return ""

        try:
            oauth = SpotifyOAuth(
                client_id=self.client_id,
                client_secret=self.client_secret,
                redirect_uri=self.redirect_uri,
                scope='playlist-modify-public playlist-modify-private user-library-read'
            )
            auth_url = oauth.get_authorize_url()
            return auth_url
        except Exception as e:
            print(f"❌ Error getting auth URL: {e}")
            return ""

    def set_oauth_token(self, code: str) -> bool:
        """Set OAuth token from authorization code"""
        if not self.use_oauth:
            return False

        try:
            oauth = SpotifyOAuth(
                client_id=self.client_id,
                client_secret=self.client_secret,
                redirect_uri=self.redirect_uri,
                scope='playlist-modify-public playlist-modify-private user-library-read'
            )
            token_info = oauth.get_access_token(code)
            if token_info:
                self.sp = spotipy.Spotify(auth=token_info['access_token'])
                return True
        except Exception as e:
            print(f"❌ Error setting OAuth token: {e}")
        return False

    def create_emotion_playlist(self, emotion: str, track_ids: List[str], playlist_name: str = None) -> Optional[str]:
        """
        Create a Spotify playlist with emotion-based recommendations

        Args:
            emotion: Detected emotion
            track_ids: List of Spotify track IDs
            playlist_name: Custom playlist name (optional)

        Returns:
            Playlist URL if successful, None otherwise
        """
        if not self.sp or not self.use_oauth:
            print("❌ OAuth authentication required for playlist creation")
            return None

        try:
            # Get current user
            user = self.sp.current_user()
            user_id = user['id']

            # Create playlist name
            if not playlist_name:
                playlist_name = f"VoiceMood AI - {emotion.title()} Vibes"

            # Create playlist
            playlist = self.sp.user_playlist_create(
                user_id,
                playlist_name,
                public=True,
                description=f"Emotion-based music recommendations for feeling {emotion}. Created by VoiceMood AI."
            )

            # Add tracks
            if track_ids:
                self.sp.playlist_add_items(playlist['id'], track_ids[:100])  # Spotify limit

            playlist_url = playlist['external_urls']['spotify']
            print(f"✅ Created playlist: {playlist_name}")
            return playlist_url

        except Exception as e:
            print(f"❌ Error creating playlist: {e}")
            return None

    def _get_search_based_recommendations(self, emotion: str, limit: int = 10) -> List[Dict]:
        """Get recommendations using Spotify search when recommendations API is unavailable"""
        try:
            # Search queries based on emotion
            search_queries = {
                'happy': ['happy upbeat pop', 'feel good music', 'cheerful songs'],
                'sad': ['sad acoustic', 'emotional ballads', 'melancholy music'],
                'angry': ['rock angry', 'intense metal', 'aggressive music'],
                'calm': ['relaxing ambient', 'peaceful classical', 'chill lo-fi'],
                'fear': ['atmospheric soundtrack', 'tense music', 'suspense'],
                'disgust': ['industrial experimental', 'noise music', 'avant-garde'],
                'neutral': ['indie alternative', 'folk music', 'indie pop'],
                'surprise': ['electronic upbeat', 'surprising music', 'unexpected hits']
            }

            queries = search_queries.get(emotion.lower(), ['pop music', 'indie songs'])
            all_tracks = []

            # Search for each query and collect tracks
            for query in queries[:2]:  # Limit to 2 queries
                results = self.sp.search(q=query, type='track', limit=limit//2 + 1)
                tracks = results['tracks']['items']
                all_tracks.extend(tracks)

            # Remove duplicates and limit results
            seen_ids = set()
            unique_tracks = []
            for track in all_tracks:
                if track['id'] not in seen_ids:
                    seen_ids.add(track['id'])
                    unique_tracks.append(track)
                if len(unique_tracks) >= limit:
                    break

            recommendations = []
            for track in unique_tracks[:limit]:
                try:
                    audio_features = self.sp.audio_features(track['id'])[0]
                except:
                    audio_features = None

                recommendation = {
                    'name': track['name'],
                    'artist': track['artists'][0]['name'],
                    'album': track['album']['name'],
                    'spotify_url': track['external_urls']['spotify'],
                    'preview_url': track.get('preview_url'),
                    'image_url': track['album']['images'][0]['url'] if track['album']['images'] else None,
                    'duration_ms': track['duration_ms'],
                    'popularity': track['popularity'],
                    'audio_features': {
                        'danceability': audio_features['danceability'] if audio_features else 0.5,
                        'energy': audio_features['energy'] if audio_features else 0.5,
                        'valence': audio_features['valence'] if audio_features else 0.5,
                        'tempo': audio_features['tempo'] if audio_features else 120,
                        'acousticness': audio_features['acousticness'] if audio_features else 0.5,
                        'instrumentalness': audio_features['instrumentalness'] if audio_features else 0.0
                    } if audio_features else {},
                    'embed_url': f"https://open.spotify.com/embed/track/{track['id']}"
                }
                recommendations.append(recommendation)

            return recommendations

        except Exception as e:
            print(f"❌ Search-based recommendations error: {e}")
            return self._get_fallback_recommendations(emotion, limit)

# Global instance
spotify_recommender = SpotifyRecommender()

def get_spotify_recommendations(emotion: str, limit: int = 5) -> List[Dict]:
    """
    Get Spotify recommendations for a detected emotion

    Args:
        emotion: Detected emotion (happy, sad, angry, etc.)
        limit: Number of recommendations

    Returns:
        List of recommended tracks with Spotify metadata
    """
    return spotify_recommender.get_recommendations(emotion, limit)

if __name__ == "__main__":
    # Test the recommender
    print("🎵 Testing Spotify Integration...")

    # Test different emotions
    emotions = ['happy', 'sad', 'angry', 'calm']

    for emotion in emotions:
        print(f"\n🎭 Getting recommendations for: {emotion}")
        recommendations = get_spotify_recommendations(emotion, 3)

        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec['name']} - {rec['artist']}")
            if 'spotify_url' in rec:
                print(f"     🔗 {rec['spotify_url']}")
            print()