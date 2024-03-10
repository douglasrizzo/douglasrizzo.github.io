---
layout: post
title: Como acessar a VPN da FEI no Linux usando o openfortivpn
categories: tutorial português linux
---

O processo para acessar a VPN da [FEI](https://fei.edu.br/) no Linux envolve o uso de um programa chamado FortiClient SSLVPN. Esse programa não é atualizado desde 2016. Os usuários de Linux têm a opção de usar o [*openfortivpn*](https://github.com/adrienverge/openfortivpn), um programa de código aberto que ainda é mantido regularmente. Este tutorial ensina como utilizá-lo.

Caso você queira fazer o procedimento original, a instituição disponibiliza um tutorial [neste link](https://fei.edu.br/vpn/VPN_FortiClient_Procedimento%20Linux.pdf). Caso contrário...

## O tutorial

- Instale o openfortivpn. Procure pelo pacote de nome `openfortivpn` no gerenciador de pacotes do seu sistema operacional. [Aqui está a lista de sistemas onde o pacote está disponível](https://github.com/adrienverge/openfortivpn#installing). Em sistemas baseados em Arch Linux:

  ```sh
  $ yay -s openfortivpn
  ```

- Crie um arquivo de configuração como o seguinte. Coloque-o onde quiser e chame-o do que quiser, e.g. `~/.config/openfortivpn/config`. `username` e `password` são suas credenciais de aluno/funcionário.

  ```ini
  host = vpn.fei.edu.br
  port = 10443
  username = uniemeusuario
  password = minhasenha
  ```

- Invoque o `openfortivpn` pelo terminal usando `sudo` e passe o caminho do arquivo de configuração criado anteriormente para a flag `-c`. Não tente usar outras flags como `-u`, pois elas não funcionam de maneira consistente.

  ```sh
  $ sudo openfortivpn -c ~/.config/openfortivpn/config
  ```

  O programa vai exibir um erro com a seguinte frase no início:

  ```
  ERROR:  Gateway certificate validation failed, and the certificate digest is not in the local whitelist. If you trust it, rerun with:
  ERROR:      --trusted-cert dabd248d06c549685cbc88ee0080c4329c6843aef8433ca223e060d73f4eba9e
  ERROR:  or add this line to your config file:
  ERROR:      trusted-cert = dabd248d06c549685cbc88ee0080c4329c6843aef8433ca223e060d73f4eba9e
  [...]
  ```

- Siga o que foi dito na mensagem de erro e adicione o parâmetro `trusted-cert` ao arquivo de configuração, que deve se parecer com isso ao final:

  ```ini
  host = vpn.fei.edu.br
  port = 10443
  username = uniemeusuario
  password = minhasenha
  trusted-cert = dabd248d06c549685cbc88ee0080c4329c6843aef8433ca223e060d73f4eba9e
  ```

- Invoque o `openfortivpn` pelo terminal novamente e o túnel será criado com sucesso:

  ```
  INFO:   Connected to gateway.
  INFO:   Authenticated.
  INFO:   Remote gateway has allocated a VPN.
  Using interface ppp0
  Connect: ppp0 <--> /dev/pts/4
  INFO:   Got addresses: [10.35.0.14], ns [172.16.0.9, 172.16.0.13]
  INFO:   Negotiation complete.
  INFO:   Negotiation complete.
  Cannot determine ethernet address for proxy ARP
  local  IP address 10.35.0.14
  remote IP address 192.0.2.1
  INFO:   Interface ppp0 is UP.
  INFO:   Setting new routes...
  INFO:   Adding VPN nameservers...
  INFO:   Tunnel is up and running.
  ```
