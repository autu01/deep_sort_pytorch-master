"""
检查当前环境中已安装的依赖包版本
"""
import sys
import subprocess

def check_package(package_name):
    """检查包是否安装及其版本"""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'show', package_name],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            version = None
            for line in lines:
                if line.startswith('Version:'):
                    version = line.split(':')[1].strip()
                    break
            return version
        else:
            return None
    except Exception as e:
        return None

def main():
    # 必需的依赖包
    required_packages = [
        'numpy',
        'scipy',
        'opencv-python',
        'Pillow',
        'easydict',
        'PyYAML',
        'tqdm',
        'torch',
        'torchvision',
        'ultralytics',
        'huggingface-hub',
        'psutil',
        'polars',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'lap',
        'motmetrics',
        'pycocotools',
        'requests',
        'python-dotenv',
        'joblib',
        'packaging',
        'Vizer'
    ]
    
    print("=" * 60)
    print("检查当前环境中已安装的依赖包")
    print("=" * 60)
    print()
    
    installed = []
    not_installed = []
    
    for package in required_packages:
        version = check_package(package)
        if version:
            installed.append((package, version))
            print(f"✅ {package:20s} {version}")
        else:
            not_installed.append(package)
            print(f"❌ {package:20s} 未安装")
    
    print()
    print("=" * 60)
    print(f"已安装: {len(installed)}/{len(required_packages)}")
    print(f"未安装: {len(not_installed)}/{len(required_packages)}")
    print("=" * 60)
    
    if not_installed:
        print("\n未安装的包:")
        for package in not_installed:
            print(f"  - {package}")
        print("\n安装命令:")
        print(f"pip install {' '.join(not_installed)}")

if __name__ == "__main__":
    main()
