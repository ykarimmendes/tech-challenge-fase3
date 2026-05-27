from langgraph_flow import executar_fluxo


if __name__ == "__main__":
    print("=== Assistente Médico Acadêmico com LangGraph ===")
    print("Digite uma pergunta clínica ou relato do paciente.")
    print("Exemplo: Paciente relata dor torácica e sudorese.\n")

    entrada = input("Entrada do paciente/profissional: ")

    resultado = executar_fluxo(entrada)

    print(resultado["resposta_final"])