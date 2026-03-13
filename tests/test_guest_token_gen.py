import time
import jwt
import uuid
from cryptography.hazmat.primitives.asymmetric import ec
from app.utils.guest_session_pool import _GUEST_SIGNING_KEY

def test_token_generation():
    user_id = str(uuid.uuid4())
    timestamp_ms = int(time.time() * 1000)
    email = f"Guest-{timestamp_ms}@guest.com"
    
    payload = {
        "id": user_id,
        "email": email
    }
    
    # Generate token
    token = jwt.encode(payload, _GUEST_SIGNING_KEY, algorithm="ES256")
    print(f"Generated Token: {token}")
    
    # Decode and verify
    # In a real scenario, we'd use the public key to verify
    public_key = _GUEST_SIGNING_KEY.public_key()
    decoded = jwt.decode(token, public_key, algorithms=["ES256"])
    
    print(f"Decoded Payload: {decoded}")
    assert decoded["id"] == user_id
    assert decoded["email"] == email
    print("Verification Successful!")

if __name__ == "__main__":
    test_token_generation()
