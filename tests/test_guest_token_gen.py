import jwt
from app.utils.guest_session_pool import _decode_token_payload

def test_token_decoding():
    # A sample payload matching the official structure
    payload = {
        "id": "9015da82-1ac5-400a-b256-ba915c3d45e1",
        "email": "Guest-1773376198189@guest.com"
    }
    
    # Generate a dummy token (HS256 is fine for testing decoding logic since we disable verification)
    token = jwt.encode(payload, "secret", algorithm="HS256")
    print(f"Sample Token: {token}")
    
    # Test decoding
    decoded = _decode_token_payload(token)
    print(f"Decoded Payload: {decoded}")
    
    assert decoded["id"] == payload["id"]
    assert decoded["email"] == payload["email"]
    print("Decoding Verification Successful!")

if __name__ == "__main__":
    test_token_decoding()
