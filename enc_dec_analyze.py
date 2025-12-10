from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import time

def encyrypt(key, message):
    if isinstance(message, str):
        message = message.encode('utf-8')
    cipher = PKCS1_OAEP.new(key)

    start_time = time.perf_counter()
    ciphertext = cipher.encrypt(message)    
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    return ciphertext, duration

def decrypt(key, ciphertext):
    cipher = PKCS1_OAEP.new(key)

    start_time = time.perf_counter()
    decrypted_data = cipher.decrypt(ciphertext)
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    return decrypted_data.decode('utf-8'), duration

if __name__ == '__main__':
   key_ids = ["key1", "key2", "key3"]

   running_times = {key_id: {"encyryption": [], "decyription": []} for key_id in key_ids}

   for key_id in key_ids:  
        with open(f'public_key_{key_id}.pem', 'rb') as f:
            public_data = f.read()
            public_key = RSA.import_key(public_data)

        with open(f'private_key_{key_id}.pem', 'rb') as f:
            private_data = f.read()
            private_data_key = RSA.import_key(private_data)