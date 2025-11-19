#!/usr/bin/env python3
"""
VoiceMood AI - Spotify Setup Helper
===================================

This script helps you set up Spotify API credentials for enhanced music recommendations.

Steps to get Spotify credentials:
1. Go to https://developer.spotify.com/dashboard
2. Click "Create an App"
3. Fill in app name and description
4. Copy Client ID and Client Secret
5. Run this script to set them up

The credentials will be saved to environment variables for the current session.
"""

import os
import sys
import getpass

def setup_spotify_credentials():
    """Interactive setup for Spotify API credentials"""
    print("🎵 VoiceMood AI - Spotify Integration Setup")
    print("=" * 50)

    print("\n📋 To get Spotify credentials:")
    print("1. Visit: https://developer.spotify.com/dashboard")
    print("2. Click 'Create an App'")
    print("3. Fill in: App name='VoiceMood AI', Description='Emotion-based music recommendations'")
    print("4. In 'Redirect URIs', add: http://127.0.0.1:8501 (for local development)")
    print("5. Copy your Client ID and Client Secret")
    print()

    # Get credentials from user
    client_id = input("Enter your Spotify Client ID: ").strip()
    client_secret = getpass.getpass("Enter your Spotify Client Secret: ").strip()

    if not client_id or not client_secret:
        print("❌ Error: Both Client ID and Client Secret are required!")
        return False

    # Set environment variables
    os.environ['SPOTIFY_CLIENT_ID'] = client_id
    os.environ['SPOTIFY_CLIENT_SECRET'] = client_secret
    os.environ['SPOTIFY_REDIRECT_URI'] = 'http://127.0.0.1:8501'  # Spotify-compliant loopback address

    # Test the connection
    try:
        from spotify_integration import SpotifyRecommender
        recommender = SpotifyRecommender(client_id, client_secret)

        if recommender.sp:
            print("✅ Spotify API connected successfully!")
            print("🎉 You can now get Spotify recommendations in VoiceMood AI!")

            # Test with a sample emotion
            print("\n🧪 Testing with 'happy' emotion...")
            recommendations = recommender.get_recommendations('happy', 3)
            if recommendations:
                print("✅ Recommendations working!")
                for rec in recommendations[:2]:
                    print(f"   🎵 {rec['name']} - {rec['artist']}")
            else:
                print("⚠️  Recommendations returned empty (this is normal for new apps)")

            return True
        else:
            print("❌ Failed to connect to Spotify API")
            print("💡 Check your credentials and try again")
            return False

    except Exception as e:
        print(f"❌ Error testing Spotify connection: {e}")
        return False

def create_env_file():
    """Create a .env file for persistent credentials"""
    if os.path.exists('.env'):
        overwrite = input(".env file already exists. Overwrite? (y/N): ").lower().strip()
        if overwrite != 'y':
            return

    try:
        with open('.env', 'w') as f:
            f.write("# VoiceMood AI - Spotify Credentials\n")
            f.write(f"SPOTIFY_CLIENT_ID={os.environ.get('SPOTIFY_CLIENT_ID', '')}\n")
            f.write(f"SPOTIFY_CLIENT_SECRET={os.environ.get('SPOTIFY_CLIENT_SECRET', '')}\n")
            f.write(f"SPOTIFY_REDIRECT_URI={os.environ.get('SPOTIFY_REDIRECT_URI', 'http://127.0.0.1:8501')}\n")
        print("✅ Credentials saved to .env file")
        print("💡 Add .env to your .gitignore to keep credentials secure!")
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")

if __name__ == "__main__":
    print(__doc__)

    success = setup_spotify_credentials()

    if success:
        save_env = input("\n💾 Save credentials to .env file for future use? (Y/n): ").lower().strip()
        if save_env != 'n':
            create_env_file()

        print("\n🚀 You're all set! Run the VoiceMood AI app:")
        print("   streamlit run predict_pytorch.py")
        print("\n🎵 Enjoy emotion-based music discovery with Spotify integration!")
    else:
        print("\n❌ Setup failed. Please check your credentials and try again.")
        print("📖 Need help? Visit: https://developer.spotify.com/documentation/web-api/")

    input("\nPress Enter to exit...")