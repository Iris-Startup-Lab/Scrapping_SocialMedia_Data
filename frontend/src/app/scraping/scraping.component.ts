/// Recursión extra en el component o en el dataservice, checar 
//import { Component } from '@angular/core';
import { Component, OnInit } from '@angular/core';
import { DataService } from '../shared/data.service';
//import { NgFor } from '@angular/common';
import { CommonModule } from '@angular/common';
//import { HttpClientModule } from '@angular/common/http';
//import { HttpClient } from '@angular/common/http';
import { FormsModule } from '@angular/forms';

console.log('ScrapingComponent loaded'); 
/*
// The next component called "ScrapingComponent" is the main component of the application.
// It is responsible for displaying the tweets obtained from the API and managing the data service.
@Component({
  selector: 'app-scraping',
  templateUrl: './scraping.component.html',
  //styleUrls: ['./scraping.component.css'],
  //imports: [],  
  //imports: [NgFor, CommonModule, HttpClientModule],
  //imports: [CommonModule, HttpClientModule],
  imports: [CommonModule],
  standalone: true,
  styles: ``
})

export class ScrapingComponent implements OnInit {
  tweets: any[] = [];
  sentimentResults: {[tweetId:string]: any}={};
  constructor(private dataService: DataService) { }
  
  ngOnInit(): void {
    console.log('ScrapingComponent initialized');
    this.dataService.getTweets().subscribe(
      (data) => {
        this.tweets = data;
        console.log('LLamando a la api para obtener Tweets recibidos:', this.tweets); 
        this.getSentimentForTweets()
      },
      (error) => {
        console.error('Error al obtener los tweets:', error);
      }
    );
  } 

  getSentimentForTweets(): void {
    const tweetIds = this.tweets.map(tweet => tweet.id);
    this.dataService.getSentimentResults(tweetIds).subscribe(
      (sentimentData) => {
        sentimentData.forEach(result => {
          console.log('Resultado del sentimiento:', result);
          this.sentimentResults[result.id] = result; // Asumiendo que el backend devuelve 'tweet' como el tweet_id
        });
        console.log('Resultados del análisis de sentimientos recibidos:', this.sentimentResults);
      },
      (error) => {
        console.error('Error al obtener los resultados del análisis de sentimientos:', error);
      }
    );
  }
}
*/

@Component({
  selector: 'app-scraping',
  templateUrl: './scraping.component.html',
  imports: [CommonModule, FormsModule],
  standalone: true,
  styles: ``
})

export class ScrapingComponent implements OnInit { // Se exporta la clase general de ScrapingComponent
  selectedSocialNetwork: string = '';
  searchQuery: string = '';
  tweets: any[] = [];
  sentimentResults: { [tweetId: string]: any } = {};
  loading: boolean = false;
  analysisResults: any = null;

  constructor(private dataService: DataService) { }

  ngOnInit(): void {
    console.log('ScrapingComponent initialized');
    this.loadInitialTweets();
  }

  loadInitialTweets(): void {
    this.loading = true;
    this.dataService.getTweets().subscribe(
      (data) => {
        this.tweets = data;
        console.log('Tweets iniciales recibidos:', this.tweets);
        this.getSentimentForTweets();
        this.loading = false;
      },
      (error) => {
        console.error('Error al obtener los tweets iniciales:', error);
        this.loading = false;
      }
    );
  }

  analyzeData(): void {
    let query = this.searchQuery.trim(); // Eliminar espacios en blanco

    if (this.selectedSocialNetwork === 'twitter' && query) {
      // Verificar si la entrada es una URL de X antes Twitter
      const urlParts = query.match(/https:\/\/(x|twitter)\.com\/[^/]+\/status\/(\d+)/);

      let tweetId: string | null = null;
      if (urlParts && urlParts[2]) {
        tweetId = urlParts[2];
        console.log('Tweet ID extraído:', tweetId);
      } else {
        console.warn('La entrada no es un enlace de tweet válido de Twitter/X.');
        this.loading = false;
        return;
      }

      this.loading = true;
      console.log(`Analizando Tweet ID: ${tweetId}`);
      this.dataService.analyzeSocialMedia({ social_network: this.selectedSocialNetwork, query: tweetId }).subscribe(
        (analysisData) => {
          this.analysisResults = analysisData;
          this.tweets = analysisData.tweets;
          this.sentimentResults = this.processSentimentResults(analysisData.sentiments);
          this.loading = false;
          console.log('Resultados del análisis recibidos:', analysisData);
        },
        (error) => {
          console.error('Error al analizar los datos:', error);
          this.loading = false;
        }
      );
    } else if (!this.selectedSocialNetwork) {
      console.warn('Por favor, selecciona una red social.');
      this.loading = false;
    } else if (!query) {
      console.warn('Por favor, ingresa un enlace de Twitter/X.');
      this.loading = false;
    }
  }

  getSentimentForTweets(): void {
    const tweetIds = this.tweets.map(tweet => tweet.id);
    this.dataService.getSentimentResults(tweetIds).subscribe(
      (sentimentData) => {
        this.sentimentResults = this.processSentimentResults(sentimentData);
        console.log('Resultados del análisis de sentimientos recibidos:', this.sentimentResults);
      },
      (error) => {
        console.error('Error al obtener los resultados del análisis de sentimientos:', error);
      }
    );
  }

  processSentimentResults(sentimentData: any[]): { [tweetId: string]: any } {
    const results: { [tweetId: string]: any } = {};
    sentimentData.forEach(result => {
      results[result.tweet_id] = result;
    });
    return results;
  }
}