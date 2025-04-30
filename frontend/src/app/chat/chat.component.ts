/*
import { Component } from '@angular/core';

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [],
  template: `
    <p>
      chat works!
    </p>
  `,
  styles: ``
})
export class ChatComponent {

}
*/

import { Component } from '@angular/core';
import { GeminiService } from '../shared/gemini.service';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
console.log('Entrando al chat component'); 

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [FormsModule, CommonModule],
  templateUrl: './chat.component.html'//,
  //styleUrls: ['./chat.component.css']
})


export class ChatComponent {
  userMessage: string = '';
  chatMessages: { sender: string; text: string }[] = [];

  constructor(private geminiService: GeminiService) { }

  sendMessage(): void {
    if (this.userMessage.trim() !== '') {
      this.chatMessages.push({ sender: 'user', text: this.userMessage });
      this.geminiService.getGeminiResponse(this.userMessage).subscribe(
        (response) => {
          this.chatMessages.push({ sender: 'bot', text: response.respuesta || response });
        },
        (error) => {
          console.error('Error al obtener respuesta de Gemini:', error);
          this.chatMessages.push({ sender: 'bot', text: 'Lo siento, no pude obtener una respuesta.' });
        }
      );
      this.userMessage = '';
    }
  }
}