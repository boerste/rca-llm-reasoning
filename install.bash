#!/bin/bash
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR=$(dirname "$SCRIPT_DIR")

print_usage() {
    echo "Usage"
    echo ""
    echo "   bash install.bash [-h|--help] [--micromamba] [--env-name=<NAME>] [--deps] [--yes]"
    echo "     -h, --help         display this usage information and exit"
    echo "     --micromamba       install micromamba (re-open the terminal after the installation)"
    echo "     --env-name=<NAME>  create the environment <NAME> (default: llama-env) with channels and Python 3.12"
    echo "     --deps             install PyTorch, CUDA 12.4, transformers, ipykernel, etc. into the environment"
    echo "     --yes              use micromamba install --yes --quiet"
    echo ""
}
if [[ -z $1 ]]; then
    print_usage
    exit
fi
ARG_MICROMAMBA=
ARG_ENV_CREATE=
ARG_DEPS=
ARG_YES=
ENV_NAME="llama-env"
for arg in "$@"; do
    case $arg in
        --micromamba)
            ARG_MICROMAMBA="true";;
        --env-name=*)
            ARG_ENV_CREATE="true"
            ENV_NAME="${arg#*=}";;
        --deps)
            ARG_DEPS="true";;
        --yes)
            ARG_YES="--yes --quiet";;
        -h | --help | *)
            print_usage
            exit;;
    esac
done

if [[ $ARG_MICROMAMBA == "true" ]]; then
    if [[ -z ${MAMBA_EXE} ]] && [[ ! -f ${HOME}/.local/bin/micromamba ]]; then
        "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
        echo "Micromamba installed to ${HOME}/micromamba"
    else
        echo "Micromamba already installed in $MAMBA_ROOT_PREFIX"
    fi
    exit 0
fi

if [[ -z ${MAMBA_EXE} ]]; then
    if [[ -f ${HOME}/.local/bin/micromamba ]]; then
        MAMBA_EXE=${HOME}/.local/bin/micromamba
        MAMBA_ROOT_PREFIX=${HOME}/micromamba
    else
        echo "MAMBA_EXE not defined and executable not found. Exiting..."
        exit 1
    fi
fi
# ensure the micromamba is available
eval "$(${MAMBA_EXE} shell hook --shell bash)"

CHANNELS="-c pytorch -c nvidia -c conda-forge"

if [[ $ARG_ENV_CREATE == "true" ]]; then
    ${MAMBA_EXE} create -n "${ENV_NAME}" ${CHANNELS} ${ARG_YES} python=3.12
fi

LLAMA_ENV_PREFIX=${MAMBA_ROOT_PREFIX}/envs/${ENV_NAME}/

if [[ $ARG_DEPS == "true" ]]; then
    ${MAMBA_EXE} -n "${ENV_NAME}" install ${ARG_YES} \
        pytorch torchaudio pytorch-cuda==12.4 cuda==12.4 \
        transformers datasets evaluate accelerate huggingface_hub \
        ipykernel ipywidgets \
        ${CHANNELS}

    # Install torchvision from pip because conda installs a CPU version and changes PyTorch to CPU as well
    ${MAMBA_EXE} -n "${ENV_NAME}" run pip install torchvision

    # Install the Jupyter Python kernel
    ${LLAMA_ENV_PREFIX}/bin/python -m ipykernel install --user --name "${ENV_NAME}" --display-name "${ENV_NAME}"
fi
