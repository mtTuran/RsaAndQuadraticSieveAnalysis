from Crypto.PublicKey import RSA
from Crypto.Util.number import inverse

if __name__ == '__main__':
    key_params = {
        "key1": {
            "p": 25117,
            "q": 25601,
            "N": 643020317,
            "e": 65537
        },
        "key2": {
            "p": 131071,
            "q": 131129,
            "N": 17187209159,
            "e": 65537
        },
        "key3": {
            "p": 262139,
            "q": 262151,
            "N": 68720000989,
            "e": 65537
        },
    }

    for key_id, params in key_params.items():
        p, q, N, e = params['p'], params['q'], params['N'], params['e']
        phi = (p - 1) * (q - 1)
        d = inverse(e, phi)

        new_rsa_key = RSA.construct((N, e, d, p, q))
        key_size = new_rsa_key.size_in_bits()

        public_rsa_key = new_rsa_key.public_key()
        public_key_pem = public_rsa_key.export_key()
        private_key_pem = new_rsa_key.export_key()

        with open(f"keys/public_key_{key_id}.pem", "wb") as f:
            f.write(public_key_pem)
        
        with open(f"keys/private_key_{key_id}.pem", "wb") as f:
            f.write(private_key_pem)

        log = f"The keys for '{key_id}' has been saved.\nKey Size: {key_size} bits.\
            \ne Size: {new_rsa_key.e.bit_length()} bit.\nd Size: {new_rsa_key.d.bit_length()} bit.\nd: {d}\n"

        print(log)
        with open("key_gen_logs.txt", "a") as f:
            f.write(log)
