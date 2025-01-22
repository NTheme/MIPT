#include <arpa/inet.h>
#include <assert.h>
#include <linux/limits.h>
#include <netinet/in.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>

typedef struct StorageItem {
  char key[PATH_MAX];
  char value[PATH_MAX];
  struct StorageItem* next;
} StorageItem;

typedef struct Storage {
  struct StorageItem* head;
} Storage;

StorageItem* find(Storage* storage, char* key) {
  StorageItem* current = storage->head;
  while (current != NULL) {
    if (strncmp(current->key, key, PATH_MAX) == 0) {
      return current;
    }
    current = current->next;
  }
  return NULL;
}

void set(Storage* storage, char* key, char* value) {
  StorageItem* element = find(storage, key);
  if (element == NULL) {
    element = malloc(sizeof(StorageItem));
    strcpy(element->key, key);
    element->next = storage->head;
    storage->head = element;
  }
  strcpy(element->value, value);
}

char* get(Storage* storage, char* key) {
  StorageItem* element = find(storage, key);
  if (element == NULL) {
    return "";
  } else {
    return element->value;
  }
}

void handle_client(int clientSocket, Storage* storage) {
  char buffer[PATH_MAX * 2];
  memset(buffer, 0, sizeof(buffer));

  ssize_t bytesRead = recv(clientSocket, buffer, sizeof(buffer), 0);
  if (bytesRead <= 0) {
    close(clientSocket);
    return;
  }

  char command[PATH_MAX];
  char key[PATH_MAX];
  char value[PATH_MAX];

  sscanf(buffer, "%s %s %[^\n]", command, key, value);

  if (strcmp(command, "get") == 0) {
    char* result = get(storage, key);
    char result_modified[PATH_MAX] = {};
    strcpy(result_modified, result);
    result_modified[strlen(result)] = '\n';
    
    send(clientSocket, result_modified, strlen(result_modified), 0);
  } else if (strcmp(command, "set") == 0) {
    set(storage, key, value);
  }
}

int main(int argc, char* argv[]) {
  assert(argc == 2 && "Invalid argument!");

  Storage* storage = malloc(sizeof(Storage));
  storage->head = NULL;

  int server_port = atoi(argv[1]);

  int server_socket = socket(AF_INET, SOCK_STREAM, 0);
  if (server_socket == -1) {
    perror("Error creating server socket");
    exit(EXIT_FAILURE);
  }

  struct sockaddr_in serverAddress;
  serverAddress.sin_family = AF_INET;
  serverAddress.sin_addr.s_addr = INADDR_ANY;
  serverAddress.sin_port = htons(server_port);

  assert(bind(server_socket, (struct sockaddr*)&serverAddress, sizeof(serverAddress)) != -1 && "Error binding socket");

  assert(listen(server_socket, 100) != -1 && "Error listening for clients");

  int epoll_fd = epoll_create1(0);
  assert(epoll_fd != -1 && "Error creating epoll descriptor");

  struct epoll_event event, events[PATH_MAX];
  event.events = EPOLLIN;
  event.data.fd = server_socket;

  assert(epoll_ctl(epoll_fd, EPOLL_CTL_ADD, server_socket, &event) != -1 && "Error adding server socket to epoll");

  while (1) {
    int nfds = epoll_wait(epoll_fd, events, PATH_MAX, -1);
    assert(nfds != -1 && "Error in epoll_wait");

    for (int i = 0; i < nfds; ++i) {
      if (events[i].data.fd == server_socket) {
        int clientSocket = accept(server_socket, NULL, NULL);
        assert(clientSocket != -1 && "Error accepting client connection");

        event.events = EPOLLIN | EPOLLET;
        event.data.fd = clientSocket;

        assert(epoll_ctl(epoll_fd, EPOLL_CTL_ADD, clientSocket, &event) != -1 && "Error adding client socket to epoll");
      } else {
        int clientSocket = events[i].data.fd;
        handle_client(clientSocket, storage);
      }
    }
  }

  close(server_socket);
  close(epoll_fd);
  return 0;
}
