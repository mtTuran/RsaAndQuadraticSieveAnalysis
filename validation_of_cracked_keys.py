from Crypto.Util.number import inverse

if __name__ == '__main__':
    key_params = {
        "key1": {
            "p": 25117,
            "q": 25601,
            "e": 65537
        },
        "key2": {
            "p": 131071,
            "q": 131129,
            "e": 65537
        },
        "key3": {
            "p": 262139,
            "q": 262151,
            "e": 65537
        },
    }

    for name, params in key_params.items():
        p = params['p']
        q = params['q']
        e = params['e']

        phi = (p - 1) * (q - 1)

        try:
            d = inverse(e, phi)
            print(f"{name} - {d}")
        except ValueError:
            print(f"{name} - Error: e is not coprime to phi.")

