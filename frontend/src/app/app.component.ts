import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
//import { AppModule } from './app.module';
import { ScrapingComponent } from './scraping/scraping.component';
import { HttpClientModule } from '@angular/common/http'


@Component({
  
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, ScrapingComponent, HttpClientModule],
  //imports: [RouterOutlet],
  templateUrl: './app.component.html'
  /*
  template: `
    <!--<h1>Welcome to {{title}}!</h1>-->
    <h1>Frontend de testeo de la App de Scraping de Iris</h1>

    <router-outlet />
  `,
  */
  //styles: []
})
export class AppComponent {
  title = 'frontend';
}


console.log('Pasó el component'); // Para depuración