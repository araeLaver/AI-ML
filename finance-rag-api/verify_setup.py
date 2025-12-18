"""
환경 설정 검증 스크립트

실행: python verify_setup.py
"""
import sys
import os

def check_python_version():
    """Python 버전 확인"""
    version = sys.version_info
    print(f"[Python] {version.major}.{version.minor}.{version.micro}")
    if version.major >= 3 and version.minor >= 10:
        print("  -> OK (3.10+ 권장)")
        return True
    else:
        print("  -> WARNING: 3.10+ 버전 권장")
        return True

def check_packages():
    """필수 패키지 설치 확인"""
    packages = {
        "openai": "LLM API 클라이언트",
        "dotenv": "환경변수 관리",
        "pydantic": "데이터 검증",
        "pydantic_settings": "설정 관리",
    }

    print("\n[패키지 설치 상태]")
    all_ok = True

    for package, desc in packages.items():
        try:
            __import__(package)
            print(f"  {package}: OK - {desc}")
        except ImportError:
            print(f"  {package}: MISSING - {desc}")
            all_ok = False

    return all_ok

def check_env_file():
    """환경 설정 파일 확인"""
    print("\n[환경 설정 파일]")

    env_exists = os.path.exists(".env")
    env_example_exists = os.path.exists(".env.example")

    print(f"  .env.example: {'OK' if env_example_exists else 'MISSING'}")
    print(f"  .env: {'OK' if env_exists else 'MISSING (생성 필요)'}")

    if not env_exists:
        print("\n  -> .env 파일 생성 방법:")
        print("     1. .env.example을 .env로 복사")
        print("     2. OPENAI_API_KEY에 실제 키 입력")

    return env_exists

def check_api_key():
    """API 키 설정 확인"""
    print("\n[API 키 상태]")

    from dotenv import load_dotenv
    load_dotenv()

    openai_key = os.getenv("OPENAI_API_KEY", "")

    if openai_key and openai_key.startswith("sk-") and len(openai_key) > 20:
        print("  OPENAI_API_KEY: OK (설정됨)")
        return True
    elif openai_key == "sk-your-openai-api-key-here":
        print("  OPENAI_API_KEY: MISSING (기본값 그대로)")
        print("\n  -> API 키 발급:")
        print("     https://platform.openai.com/api-keys")
        return False
    else:
        print("  OPENAI_API_KEY: MISSING")
        print("\n  -> API 키 발급:")
        print("     https://platform.openai.com/api-keys")
        return False

def check_project_structure():
    """프로젝트 구조 확인"""
    print("\n[프로젝트 구조]")

    required_dirs = [
        "src",
        "src/api",
        "src/rag",
        "src/core",
        "tests",
        "notebooks",
        "data",
        "data/documents",
    ]

    required_files = [
        "requirements.txt",
        ".env.example",
        ".gitignore",
        "src/core/config.py",
        "notebooks/01_llm_api_basics.py",
    ]

    all_ok = True

    for dir_path in required_dirs:
        exists = os.path.isdir(dir_path)
        status = "OK" if exists else "MISSING"
        print(f"  {dir_path}/: {status}")
        if not exists:
            all_ok = False

    print()
    for file_path in required_files:
        exists = os.path.isfile(file_path)
        status = "OK" if exists else "MISSING"
        print(f"  {file_path}: {status}")
        if not exists:
            all_ok = False

    return all_ok

def main():
    print("=" * 60)
    print("  Finance RAG API - 환경 설정 검증")
    print("=" * 60)

    results = []

    results.append(("Python 버전", check_python_version()))
    results.append(("패키지 설치", check_packages()))
    results.append(("프로젝트 구조", check_project_structure()))
    results.append(("환경 파일", check_env_file()))
    results.append(("API 키", check_api_key()))

    print("\n" + "=" * 60)
    print("  검증 결과")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("모든 검증 통과! 실습을 시작할 수 있습니다.")
        print("\n다음 명령어로 첫 번째 실습 시작:")
        print("  python notebooks/01_llm_api_basics.py")
    else:
        print("일부 항목이 설정되지 않았습니다.")
        print("위의 안내를 따라 설정을 완료하세요.")

    return 0 if all_passed else 1

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
