import time
import os
from Crypto.PublicKey import RSA
from Crypto.Util.number import bytes_to_long

CHUNK_SIZE = 3 
BLOCK_SIZE = 5 
NUMBER_OF_TESTS_SHORT_MESSAGES = 1000
NUMBER_OF_TESTS_LONG_MESSAGES = 100

def encrypt(key, message_str):
    if isinstance(message_str, str):
        message_bytes = message_str.encode('utf-8')
    
    m_int = bytes_to_long(message_bytes)
    
    start_time = time.perf_counter()
    ciphertext = pow(m_int, key.e, key.n)
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    return ciphertext, duration

def decrypt(key, ciphertext):
    start_time = time.perf_counter()
    message = pow(ciphertext, key.d, key.n)
    end_time = time.perf_counter()

    message_str = message.to_bytes(CHUNK_SIZE, byteorder='big').decode('utf-8')
    duration = end_time - start_time
    return message_str, duration

def encrypt_by_chunking(key, input_path, output_path):
    start_time = time.perf_counter()
    
    with open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
        while True:
            chunk = f_in.read(CHUNK_SIZE)
            if not chunk:
                break
            
            message = bytes_to_long(chunk)
            ciphertext = pow(message, key.e, key.n)            
            f_out.write(ciphertext.to_bytes(BLOCK_SIZE, byteorder='big'))
            
    end_time = time.perf_counter()
    return end_time - start_time

def decrypt_by_chunking(key, input_path, output_path):
    start_time = time.perf_counter()
    
    with open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
        while True:
            enc_block = f_in.read(BLOCK_SIZE)
            if not enc_block:
                break
            
            ciphertext = int.from_bytes(enc_block, byteorder='big')
            message = pow(ciphertext, key.d, key.n)
            decrypted_chunk = message.to_bytes(CHUNK_SIZE, byteorder='big')
            f_out.write(decrypted_chunk)
            
    end_time = time.perf_counter()
    return end_time - start_time


if __name__ == '__main__':
    pdf_filename = "message.pdf"  
    file_size = os.path.getsize(pdf_filename)
    log_path = "enc_dec_analayze_logs.txt"

    log = f"Using Chunk Size: {CHUNK_SIZE}\nUsing Block Size: {BLOCK_SIZE}\n-----------------------\n"
    with open(log_path, "w") as f:
        f.write(log)

    keys_path = "keys/"
    key_ids = ["key1", "key2", "key3"] 

    running_times = {
        key_id: {
            "short_msg_enc": 0, 
            "short_msg_dec": 0,
            "pdf_enc": 0,
            "pdf_dec": 0
        } 
        for key_id in key_ids
    }

    # Running time computation for messages that do not require chunking
    for key_id in key_ids:
        print(f"--- Analyzing for short messages: {key_id} ---")

        with open(f'{keys_path}public_key_{key_id}.pem', 'rb') as f:
            public_key = RSA.import_key(f.read())
        with open(f'{keys_path}private_key_{key_id}.pem', 'rb') as f:
            private_key = RSA.import_key(f.read())
  
        short_msg = "Oki"

        for _ in range(NUMBER_OF_TESTS_SHORT_MESSAGES):
            ciphertext, running_time = encrypt(public_key, short_msg)
            running_times[key_id]["short_msg_enc"] = running_times[key_id]["short_msg_enc"] + running_time
            
            decrypted_message, running_time = decrypt(private_key, ciphertext)
            running_times[key_id]["short_msg_dec"] = running_times[key_id]["short_msg_dec"] + running_time
        
        avg_enc_time = running_times[key_id]["short_msg_enc"] / NUMBER_OF_TESTS_SHORT_MESSAGES
        avg_dec_time = running_times[key_id]["short_msg_dec"] / NUMBER_OF_TESTS_SHORT_MESSAGES
        
        log = log = (
            f"Key ID: {key_id}\n"
            f"Message Length: {len(short_msg)} bytes\n"
            f"Number of tests: {NUMBER_OF_TESTS_SHORT_MESSAGES}\n"
            f"Average Encryption Time: {avg_enc_time:.6f} s\n"
            f"Average Decryption Time: {avg_dec_time:.6f} s\n"
            f"\n-----------------------\n"
        )
        
        print(log)
        with open(log_path, "a") as f:
            f.write(log)

    # Running time computation for messages that require chunking
    for key_id in key_ids:
        print(f"--- Analyzing for long messages: {key_id} ---")

        with open(f'{keys_path}public_key_{key_id}.pem', 'rb') as f:
            public_key = RSA.import_key(f.read())
        with open(f'{keys_path}private_key_{key_id}.pem', 'rb') as f:
            private_key = RSA.import_key(f.read())
  
        enc_pdf_name = f"enc_{key_id}.bin"
        dec_pdf_name = f"dec_{key_id}.pdf"

        for _ in range(NUMBER_OF_TESTS_LONG_MESSAGES):
            running_time = encrypt_by_chunking(public_key, pdf_filename, enc_pdf_name)
            running_times[key_id]["pdf_enc"] = running_times[key_id]["pdf_enc"] + running_time
            
            running_time = decrypt_by_chunking(private_key, enc_pdf_name, dec_pdf_name)
            running_times[key_id]["pdf_dec"] = running_times[key_id]["pdf_dec"] + running_time

        avg_enc_time = running_times[key_id]["pdf_enc"] / NUMBER_OF_TESTS_LONG_MESSAGES
        avg_dec_time = running_times[key_id]["pdf_dec"] / NUMBER_OF_TESTS_LONG_MESSAGES

        log = log = (
            f"Key ID: {key_id}\n"
            f"Message Length: {file_size} bytes\n"
            f"Number of tests: {NUMBER_OF_TESTS_LONG_MESSAGES}\n"
            f"Average Encryption Time: {avg_enc_time:.6f} s\n"
            f"Average Decryption Time: {avg_dec_time:.6f} s\n"
            f"\n-----------------------\n"
        )
    
        print(log)
        with open(log_path, "a") as f:
            f.write(log)
        
